#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import pdb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from moco.dataloader import *
from moco.builder import AdaptiveConcatPool2d, create_head1d
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import math
from cluster_logging import export_results

from itertools import chain
def get_paramgroups(model,args,resnet=True):
    #https://discuss.pytorch.org/t/implementing-differential-learning-rate-by-parameter-groups/32903
    if(resnet):
        pgs = [[model.conv1,model.bn1,model.layer1,model.layer2], [model.layer3,model.layer4],[model.fc]]
    else:#densenet
        pgs = [[model.features],[model.classifier]]
    pgs = [[p.parameters() for p in pg] for pg in pgs]
    lgs = [{"params":chain(*pg), "lr":args.lr/pow(10,len(pgs)-1-i)} for i,pg in enumerate(pgs)]
    return lgs


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')#was 0.
parser.add_argument('-p', '--print-freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on test set (and export predictions)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')


##############################
#extra arguments introduced by us
parser.add_argument('--dataset', default='imagenet', help='imagenet/chexpert(14)(_eval)/chexphoto(14)(_eval)/mimic_cxr(_cxr14)/cxr14/diabetic_retinopathy/cifar10/stl10(_x)')
parser.add_argument('--eval_dataset', default='imagenet', help='messidor1/messidor2')
parser.add_argument('--optimizer', default='adam', help='sgd/adam(w)')#was sgd
parser.add_argument('--unfreeze', action='store_true',
                    help='unfreeze all layers')
parser.add_argument('--discriminative-lrs', action='store_true',dest="discriminative_lrs",
                    help='discriminative i.e. parameter-group-dependent learning rates')
parser.add_argument('--training-fraction', default=1.0, type=float,dest="training_fraction",
                    help='fraction of training examples to be used.')
parser.add_argument('--output-path', default='.', type=str,dest="output_path",
                    help='output path')
parser.add_argument('--code-path', default='/opt/submit/', type=str, dest="code_path",
                    help='code path')
parser.add_argument('--save-freq', default=100, type=int,
                    metavar='N', help='save frequency (default: 100)', dest="save_freq")
parser.add_argument('--num_classes', default=5, type=int, metavar='M',
                    help='num_classes')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--metadata', default='', type=str,
                    help='metadata for output')
parser.add_argument('--pretrained-backbone', default='', type=str, dest='pretrained_backbone',
                    help='path to pretrained checkpoint/state dict from earlier runs/imagenet pretraining')
parser.add_argument('--imagenet-stats', action='store_true', dest='imagenet_stats',
                    help='use imagenet stats instead of dataset stats (use with pretrained imagenet models)')
parser.add_argument('--image-size', default=224, type=int, dest='image_size',
                    help='image size in pixels')
parser.add_argument('--add-validation-set', action='store_true', dest='add_validation_set',
                    help='split off validation set')
parser.add_argument('--custom-split', action='store_true', dest='custom_split',
                    help='custom stratified split 80:10:10')
parser.add_argument('--random-split', action='store_true', dest='random_split',
                    help='random split diregarding patients 80:10:10')                    
parser.add_argument('--bootstrap-samples', default=0, type=int, dest='bootstrap_samples',
                    help='number of bootstrap samples during evaluation (0 for no bootstrap)')
parser.add_argument('--test-every', default=1, type=int,
                    metavar='N', help='test and save every epoch', dest="test_every")
parser.add_argument('--eval-dr', default='quarternary', type=str, dest='eval_dr',
                    help='binary/ternary/quarternary/quinary -- 2/3/4/5')
parser.add_argument('--eval-criteria', default='binary_rdr', type=str, dest='eval_criteria',
                    help='binary_ -- rdr/dme/norm')
parser.add_argument('--add-mc-dropout', action='store_true', dest='add_mc_dropout',
                    help='add dropout layers')
parser.add_argument('--mixup', action='store_true', help='use mixup (alpha=1) during training')
parser.add_argument('--lin-ftrs', type=str, default=None, help='linear filters for head (as string e.g. [1024]) overrides --add-mc-dropout')
parser.add_argument('--update-bn-stats', action='store_true', help='update bn-stats also in frozen state (when training head only)')

best_acc1 = 0


def main():
    args = parser.parse_args()
    args.executable = "main_lincls"

    if args.dataset == 'diabetic_retinopathy':
        args.output_path = args.output_path+"/"+str(args.training_fraction)+"/"

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    print(args)
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)    

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()

def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

#from https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.dataset == "imagenet":
        num_classes = 1000
    elif args.dataset.startswith("chexpert14") or args.dataset.startswith("chexphoto14") or args.dataset.startswith("mimic_cxr") or args.dataset.startswith("cxr14"):
        num_classes = 14
    elif args.dataset.startswith("chexpert") or args.dataset.startswith("chexphoto"):
        num_classes = 5
    elif args.dataset == "diabetic_retinopathy":
        num_classes = args.num_classes
    elif args.dataset == "cifar10" or args.dataset.startswith("stl10"):
        num_classes = 10
    else:
        assert(False)
        
    model = models.__dict__[args.arch](num_classes=num_classes)
    is_resnet = hasattr(model,"fc") #distinguish resnet vs densenet

    #reduced stem for small datasets such as cifar (hack for resnet models)
    if(args.dataset=="cifar10"):
        model.conv1= torch.nn.Conv2d(3,model.conv1.out_channels,kernel_size = (3,3),stride=(1,1),padding=(1,1),bias=False)
        model.max_pool = torch.nn.Identity()
    if(args.lin_ftrs is not None):
        args.lin_ftrs=eval(args.lin_ftrs)
        concat_pooling = True #only possible for resnet at the moment
        if(concat_pooling and is_resnet):
            model.avgpool = AdaptiveConcatPool2d()
        model.fc = create_head1d(2048 if is_resnet else 1024,num_classes,ps=0.5,concat_pooling=concat_pooling and is_resnet)
       

    if args.add_mc_dropout:
        if(is_resnet):#resnet
            model.fc = nn.Sequential(nn.Dropout(p=0.2),
                                     nn.Linear(2048,1024),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(1024,num_classes))
        else:
            model.classifier = nn.Sequential(nn.Dropout(p=0.2),
                                             nn.Linear(1024,1024),
                                             nn.ReLU(),
                                             nn.Dropout(p=0.2),
                                             nn.Linear(1024,num_classes))


    if not args.unfreeze:
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if not(name.startswith("fc.")) and not(name.startswith("classifier.")):
                param.requires_grad = False

    #load e.g. imagenet pretrained model or model from other run
    if args.pretrained_backbone:
        if os.path.isfile(args.pretrained_backbone):
            print("=> loading pretrained base '{}'".format(args.pretrained_backbone))
            state_dict = torch.load(args.pretrained_backbone, map_location="cpu")
            if("state_dict" in list(state_dict.keys())):
                state_dict = state_dict["state_dict"]
            state_dict_classes = state_dict["fc.bias"].size()[0] if is_resnet else state_dict["classifier.bias"].size()[0]
            
            if(state_dict_classes != num_classes):
                print("number of classes in state dict (",state_dict_classes,") do not match ",num_classes,". Discarding fc layer.")
                if(is_resnet):
                    del state_dict["fc.bias"]
                    del state_dict["fc.weight"]
                else:
                    del state_dict["classifier.bias"]
                    del state_dict["classifier.weight"]
            model.load_state_dict(state_dict, strict=False)
            if(state_dict_classes != num_classes):
                # init the fc layer
                if(is_resnet):#resnet
                    model.fc.apply(init_weights)
                else:
                    model.classifier.apply(init_weights)
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained_backbone))
    else:
        # init the fc layer
        if(is_resnet):#resnet
            model.fc.apply(init_weights)
        else:
            model.classifier.apply(init_weights)
    
    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pre-trained model from checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc') and not k.startswith('module.encoder_q.classifier'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                #semi-supervised pretraining (keep the semi-supervised head)
                if(k=="module.fc_ss"):
                    if(is_resnet):#resnet
                        state_dict["module.fc"]=state_dict[k]
                    else:#densenet
                        state_dict["module.classifier"]=state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            #assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no pre-trained model checkpoint found at '{}'".format(args.pretrained))

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    if(args.dataset.startswith("chexpert") or args.dataset.startswith("chexphoto") or args.dataset.startswith("mimic_cxr") or args.dataset.startswith("cxr14")):#multi-label
        criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.unfreeze:
        if args.discriminative_lrs:
            #assert(isinstance(model.module,models.resnet.ResNet))
            parameters = get_paramgroups(model.module if args.gpu is None else model,args,resnet=is_resnet)
        else:
            parameters = model.parameters()
    else:
    # optimize only the linear classifier
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        if args.add_mc_dropout:
            assert len(parameters) == 4  # fc.weight, fc.bias
        elif args.lin_ftrs is None: 
            assert len(parameters) == 2  # fc.weight, fc.bias
        
    if(args.optimizer == "sgd"):
        optimizer = torch.optim.SGD(parameters, args.lr,
                                    momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif(args.optimizer == "adam"):
        optimizer = torch.optim.AdamW(parameters, args.lr,
                                    betas=(args.momentum,0.999),
                                weight_decay=args.weight_decay)
    else:
        assert(False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1#.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            if not args.evaluate:
                optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    #prepare stats for normalization
    if(args.dataset=="imagenet" or args.imagenet_stats or args.dataset.startswith("stl10")):
        normalize = transforms.Normalize(mean=imagenet_stats[0],
                                     std=imagenet_stats[1])
    elif(args.dataset.startswith("chexpert") or args.dataset.startswith("chexphoto") or args.dataset.startswith("mimic_cxr") or args.dataset.startswith("cxr14")):
        normalize = transforms.Normalize(mean=chexpert_stats[0],
                                     std=chexpert_stats[1])
    elif(args.dataset == "diabetic_retinopathy"):
        normalize = transforms.Normalize(mean=dr_stats[0],
                                     std=dr_stats[1])
    elif(args.dataset=="cifar10"):
        normalize = transforms.Normalize(mean=cifar_stats[0],
                                     std=cifar_stats[1])                            
    # Data loading code
    if(args.dataset=="imagenet" or args.dataset.startswith("stl10")):
        train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            
        test_transforms = transforms.Compose([
            transforms.Resize(int(1.14286*args.image_size)), # was 256
            transforms.CenterCrop(args.image_size), # was 224
            transforms.ToTensor(),
            normalize,
        ])
        if(args.dataset == "imagenet"):
            traindir = os.path.join(args.data, 'train')
            testdir = os.path.join(args.data, 'val')
            df_train, label_itos = prepare_imagefolder_df(traindir)
            df_test, _ = prepare_imagefolder_df(testdir,label_itos)
        
    elif(args.dataset.startswith("chexpert") or args.dataset.startswith("chexphoto") or args.dataset.startswith("mimic_cxr") or args.dataset.startswith("cxr14")):
        #these are the "Comparison of Deep Learning Approaches... transformations
        train_transforms = transforms.Compose([
                transforms.RandomRotation(10 if args.dataset.startswith("cxr14") else 15),#was 7 in the original reference- these are the values from CheXclusion
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transforms = transforms.Compose([
            transforms.Resize(int(1.14286*args.image_size)), # was 256
            transforms.CenterCrop(args.image_size), # was 224
            transforms.ToTensor(),
            normalize,
        ])
        
        if(args.dataset.startswith("chexpert") or args.dataset.startswith("chexphoto")):
            df_train, df_test, label_itos = prepare_chexpert_df(args.data,label_itos=label_itos_chexpert14 if (args.dataset=="chexpert14" or args.dataset =="chexphoto14") else label_itos_chexpert5,chexphoto=args.dataset.startswith("chexphoto"),full_dataset_as_test=args.dataset.endswith("eval"))
            df_val = None
        elif(args.dataset.startswith("mimic_cxr")):
            df_train, df_val, df_test, label_itos = prepare_mimic_cxr_df(args.data,use_chexpert_labels=not(args.dataset.endswith("cxr14")))
            args.add_validation_set = True
        elif(args.dataset.startswith("cxr14")):
            df_train, df_test, label_itos = prepare_cxr14_df(args.data)
            df_val = None
        #df_train["label"]=df_train["label_raw"]#could also use label_raw and some random target_transform in the dataloader as in 1911.06475
        #df_test["label"]=df_valid["label_raw"]
        
    elif(args.dataset == "diabetic_retinopathy"):
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.ToTensor(),
            normalize
        ])
        test_transforms = transforms.Compose([
            transforms.Resize(int(1.14286*args.image_size)), # was 256
            transforms.CenterCrop(args.image_size), # was 224
            transforms.ToTensor(),
            normalize,
        ])

        if args.eval_dataset == 'messidor1':
            df_train, df_test, label_itos = prepare_messidor_1_df(args.data)
        elif args.eval_dataset == 'messidor2':
            df_train, df_test, label_itos = prepare_messidor_df(args.data)
        else:
            if not args.evaluate:
                df_train, df_test, _, label_itos = prepare_diabetic_retinopathy_df(args.data, args)
            else:
                df_train, _, df_test, label_itos = prepare_diabetic_retinopathy_df(args.data, args)
                
        #args.add_validation_set = True
        df_val = None
        #pdb.set_trace()
        df_train = prepare_binary_classification(df_train, args.eval_criteria)
        #df_val = prepare_binary_classification(df_val, args.eval_criteria)
        df_test = prepare_binary_classification(df_test, args.eval_criteria)
        
    elif args.dataset == "cifar10":
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        
        traindir = os.path.join(args.data, 'train')
        testdir = os.path.join(args.data, 'test')
        df_train, label_itos = prepare_imagefolder_df(traindir)
        df_test, _ = prepare_imagefolder_df(testdir,label_itos)
        df_val = None

    #pdb.set_trace()
    if(args.custom_split or args.random_split):
        df_all = pd.concat([df_train,df_val,df_test] if df_val is not None else [df_train,df_test])
        if(args.distributed is False or args.rank==0):#only create splits for rank0
            split_stratified_wrapper(df_all,fractions=[0.8,0.1,0.1],dataset=args.dataset,save_path=os.path.join(args.code_path,"moco_official/dataset_splits/"),col_subset="custom_split",filename_postfix="custom" if args.custom_split else "random2",disregard_patients=False,disregard_labels=args.random_split)
        if(args.distributed):
            dist.barrier()#wait until rank0 is finished
        df_all = split_stratified_wrapper(df_all,fractions=[0.8,0.1,0.1],dataset=args.dataset,save_path=os.path.join(args.code_path,"moco_official/dataset_splits/"),col_subset="custom_split",filename_postfix="custom" if args.custom_split else "random2",disregard_patients=False,disregard_labels=args.random_split)#load split with all ranks
        df_train = df_all[df_all.custom_split==0].copy()
        df_val = df_all[df_all.custom_split==1].copy()
        df_test = df_all[df_all.custom_split==2].copy()
        args.add_validation_set = True

    elif(args.add_validation_set and not(args.dataset.startswith("mimic_cxr")) and args.dataset !="diabetic_retinopathy"): #split off validation set if desired (for consistency with downstream training)
        if(args.distributed is False or args.rank==0):#only create splits for rank0
            split_stratified_wrapper(df_train,fractions=[0.9,0.1],dataset=args.dataset,save_path=os.path.join(args.code_path,"moco_official/dataset_splits/"),col_subset="val",filename_postfix="val")
        if(args.distributed):
            dist.barrier()#wait until rank0 is finished
        df_train = split_stratified_wrapper(df_train,fractions=[0.9,0.1],dataset=args.dataset,save_path=os.path.join(args.code_path,"moco_official/dataset_splits/"),col_subset="val",filename_postfix="val")#load split with all ranks
        df_train_new = df_train[df_train.val==0].copy()
        df_val = df_train[df_train.val==1].copy()
        df_train = df_train_new

    if(args.training_fraction<1.0 and not args.evaluate): #reduced training set
        filename_postfix = "custom_" if args.custom_split else ("wo_val_" if args.add_validation_set else "")
        filename_postfix +="subset_"+str(args.training_fraction)

        if(args.distributed is False or args.rank==0):#only create splits for rank0
            split_stratified_wrapper(df_train,fractions=[1-args.training_fraction,args.training_fraction],dataset=args.dataset,save_path=os.path.join(args.code_path,"moco_official/dataset_splits/"),col_subset="subset",filename_postfix=filename_postfix)
        if(args.distributed):
            dist.barrier()#wait until rank0 is finished
        df_train = split_stratified_wrapper(df_train,fractions=[1-args.training_fraction,args.training_fraction],dataset=args.dataset,save_path=os.path.join(args.code_path,"moco_official/dataset_splits/"),col_subset="subset",filename_postfix=filename_postfix)
        df_train = df_train[df_train.subset==1].copy() #remove the unlabelled samples for now (no semi-supervised training)
        #if(args.training_fraction<1.0 and args.dataset=="diabetic_retinopathy"): #reduced training set
            
    #create the actual dataset
    if(args.dataset.startswith("stl10")):
        assert(args.image_size<=96)
        fold = args.dataset.split("_")
        fold = int(fold[1]) if(len(fold)==2) else None
        label_itos = label_itos_stl10

        train_dataset = datasets.STL10(args.data, 
            split='train', 
            folds=fold, 
            transform=train_transforms)
        test_dataset = datasets.STL10(args.data, 
            split='test', 
            folds=fold, 
            transform=train_transforms)
    else:
        train_dataset = ImageDataframeDataset(df_train, train_transforms)
        test_dataset = ImageDataframeDataset(df_test, test_transforms)
        if(args.add_validation_set or args.custom_split):
            val_dataset = ImageDataframeDataset(df_val, test_transforms)    

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    if(args.add_validation_set):
        val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:#evaluation only
        if(args.add_validation_set):
            acc1_val = validate_full(val_loader, model, criterion, args, "val", bootstrap_samples = args.bootstrap_samples)    
        acc1_test = validate_full(test_loader, model, criterion, args, "test", bootstrap_samples = args.bootstrap_samples, export_preds=True)
        #write to file
        results_dict = {"result_best_acc1_test":str(acc1_test)}
        if(args.add_validation_set):
            results_dict["result_best_acc1_val"] = str(acc1_val)
        export_results([args,results_dict],os.path.join(args.output_path,"results.json"))
        return

    history_acc1_val=[]        
    history_acc1_test=[]
    best_acc1_val=0
    best_acc1_val_epoch=0
    best_acc1_test=0
    best_acc1_test_epoch=0
    best_acc1_test_at_best_acc1_val=0

    for epoch in range(args.start_epoch, args.epochs):
      if args.distributed:
          train_sampler.set_epoch(epoch)
      adjust_learning_rate(optimizer, epoch, args)

      # train for one epoch
      train(train_loader, model, criterion, optimizer, epoch, args)

      if epoch % args.test_every == 0:   
        # evaluate on test set
        acc1_test = validate_full(test_loader, model, criterion, args, "test epoch "+str(epoch), bootstrap_samples = args.bootstrap_samples)
        history_acc1_test.append(acc1_test)
        
        # evaluate on validation set
        if(args.add_validation_set):
            acc1_val = validate_full(val_loader, model, criterion, args, "val epoch "+str(epoch), bootstrap_samples = args.bootstrap_samples)
            history_acc1_val.append(acc1_val)
            is_best = acc1_val > best_acc1_val
            if(is_best):
                best_acc1_val = acc1_val
                best_acc1_val_epoch = epoch
                best_acc1_test_at_best_acc1_val=acc1_test        
        else:
            # remember best acc@1 and save checkpoint
            is_best = acc1_test > best_acc1_test

        if(acc1_test > best_acc1_test):
            best_acc1_test = acc1_test
            best_acc1_test_epoch = epoch
            
        #write results file (in case the process ends)
        results_dict = {"result_best_acc1_test":str(best_acc1_test), "result_history_acc1_test":[*map(str,history_acc1_test)],"result_best_acc1_test_epoch":str(best_acc1_test_epoch)}
        if(args.add_validation_set):
            results_dict["result_best_acc1_val"] = str(best_acc1_val)
            results_dict["result_best_acc1_test_at_best_acc1_val"] = str(best_acc1_test_at_best_acc1_val)
            results_dict["result_history_acc1_val"] = [*map(str,history_acc1_val)]
            results_dict["result_best_acc1_val_epoch"] = str(best_acc1_val_epoch)
            
        export_results([args,results_dict],os.path.join(args.output_path,"results.json"))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            if(is_best or (epoch +1)%args.save_freq ==0 or epoch+1 == args.epochs):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, os.path.join(args.output_path,'checkpoint_latest.pth.tar'))#'checkpoint_{}.pth.tar'.format(epoch)
            if not args.unfreeze:
                if epoch == args.start_epoch:
                    #sanity_check(model.state_dict(), args.pretrained)
                    print("No sanity check for now")

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    
    if args.dataset.startswith("chexpert") or args.dataset.startswith("chexphoto") or args.dataset.startswith("mimic_cxr") or args.dataset.startswith("cxr14"):
        auc = AverageMeter('macro AUC', ':6.2f')
        metrics = [auc]
    else:
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        auc = AverageMeter('macro AUC', ':6.2f')
        metrics = [top1,auc]
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, *metrics],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    if(args.unfreeze or args.update_bn_stats):
        model.train()
    else:
        model.eval()
        
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #if i > 5: break
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        if(args.mixup):
             images, target_a, target_b, lam = mixup_data(images, target, alpha=1)
             output = model(images)
             loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            # compute output
            output = model(images)
            loss = criterion(output, target)
        
        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))
        
        if(args.dataset.startswith("chexpert") or args.dataset.startswith("chexphoto") or args.dataset.startswith("mimic_cxr") or args.dataset.startswith("cxr14")):
            auc1 = macro_auc(output, target)
            if(auc1>0):#otherwise auc could not be evaluated
                auc.update(auc1, images.size(0))
        else:
            if args.eval_criteria == "quinary":
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                psuedo_target=target.clone()
                psuedo_target[target>=1]=1
                auc1 = macro_auc(nn.Softmax(dim=1)(output)[:,1:].sum(1),psuedo_target)
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 1))
                auc1 = macro_auc(output[:,1],target)

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            auc.update(auc1, images.size(0))
            
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.print_freq>0 and i % args.print_freq == 0:
            progress.display(i)
        
def validate_full(val_loader, model, criterion, args, output_prefix="", bootstrap_samples=0, bootstrap_alpha=0.95, export_preds=False):
    '''evaluate on full output rather than batch-wise'''
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    
    # switch to evaluate mode
    model.eval()
    if args.add_mc_dropout and args.evaluate:
        model.apply(apply_dropout)
        epochs = 10 #number of mc-samples
    else:
        epochs = 1
    with torch.no_grad():
        end = time.time()
        all_outputs = []
        for epoch in range(epochs):
          targets,outputs = [],[]  
          for i, (images, target) in enumerate(val_loader):
            
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            if 'messidor' in args.eval_dataset:
                target = target.long()
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # store outputs and targets
            losses.update(loss.item(), images.size(0))
            outputs.append(output.detach().cpu())
            targets.append(target.cpu())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
          all_outputs.append(outputs)
          #pdb.set_trace()
          if args.dataset == "diabetic_retinopathy":
              acc1, auc = accuracy_dr(torch.cat(outputs).clone(), torch.cat(targets).clone(), args, topk=(1, 1), eval_dr=args.eval_criteria)
              print("Epoch: {} Evaluation: {} Acc1: {} Auc: {} ".format(epoch, args.eval_criteria, acc1[0], auc))
              np.save(os.path.join(args.output_path,"{}_{}_{}_preds.npy".format(args.eval_dataset, epoch, 'test')),torch.cat(outputs).cpu().numpy())
                
        if args.add_mc_dropout and args.evaluate:
            all_outputs = torch.stack([torch.cat(alp) for alp in all_outputs])
            outputs = all_outputs.mean(0)
            outputs_std = all_outputs.std(0)
            outputs_entropy = torch.sum(-all_outputs*torch.log(all_outputs+1e-5),dim=0)            
        else:
            outputs = torch.cat(outputs)
        targets = torch.cat(targets)

        if(export_preds):
            if args.evaluate:
                np.save(os.path.join(args.output_path,"{}_{}_preds.npy".format(args.eval_dataset, 'test')),outputs.cpu().numpy())
                if args.add_mc_dropout:
                    np.save(os.path.join(args.output_path,"{}_{}_std_preds.npy".format(args.eval_dataset, 'test')),outputs_std.cpu().numpy())
                    np.save(os.path.join(args.output_path,"{}_{}_entropy_preds.npy".format(args.eval_dataset, 'test')),outputs_entropy.cpu().numpy())
                np.save(os.path.join(args.output_path,"{}_{}_targs.npy".format(args.eval_dataset, 'test')),targets.cpu().numpy())
            else:
                np.save(os.path.join(args.output_path,"{}_{}_preds.npy".format(args.eval_dataset, 'val')),outputs.cpu().numpy())
                np.save(os.path.join(args.output_path,"{}_{}_targs.npy".format(args.eval_dataset, 'val')),targets.cpu().numpy())
            
        if args.dataset.startswith("chexpert") or args.dataset.startswith("chexphoto") or args.dataset.startswith("mimic_cxr") or args.dataset.startswith("cxr14"):
            auc = macro_auc(outputs,targets)
            aucs = label_aucs(outputs,targets)
            
            if(bootstrap_samples>0):
                np.random.seed(42)
                bootstrap_ids = [np.random.choice(len(outputs),replace=True,size=len(outputs)) for _ in range(bootstrap_samples)]
                auc_diff = [macro_auc(outputs[b],targets[b])-auc for b in bootstrap_ids]
                auc_low = auc + np.percentile(auc_diff, ((1.0-bootstrap_alpha)/2.0) * 100)
                auc_high = auc + np.percentile(auc_diff, (bootstrap_alpha+((1.0-bootstrap_alpha)/2.0)) * 100)
                
                aucs_diff = [label_aucs(outputs[b],targets[b])-aucs for b in bootstrap_ids]
                aucs_low = aucs + np.percentile(aucs_diff, ((1.0-bootstrap_alpha)/2.0) * 100,axis=0)
                aucs_high = aucs + np.percentile(aucs_diff, (bootstrap_alpha+((1.0-bootstrap_alpha)/2.0)) * 100,axis=0)
                print(output_prefix+' bootstrap(95): macro AUC (low,pt,high) ',auc_low,auc,auc_high, 'label AUCs (low,pt,high)',aucs_low,aucs,aucs_high)
            else:
                print(output_prefix+': macro AUC ',auc, "label AUCs",aucs) 
            return auc
        else:
            if args.dataset == "diabetic_retinopathy":
                if args.eval_criteria == 'quinary': 
                    evaluations = ["quinary"]#+[ "quarternary", "ternary","binary_dme", "binary_rdr", "binary_norm"] #[::-1]
                elif args.eval_criteria == 'binary_rdr': 
                    evaluations = [ "binary_rdr"] #[::-1]
                elif args.eval_criteria == 'binary_norm': 
                    evaluations = [ "binary_norm"] #[::-1]
                elif args.eval_criteria == 'binary_dme': 
                    evaluations = [ "binary_dme"] #[::-1]
                
                accs = []
                aucs = [] 
                if args.evaluate:
                    filename = args.output_path+args.eval_dataset+"_test_results.npy"
                else:
                    filename = args.output_path+args.eval_dataset+"_val_results.npy"
                print(filename)
                save_results = {}
                
                # np.save('results/{}_test_results_check.npy'.format(args.training_fraction),targets.detach().cpu().numpy())
                # import sys; sys.exit()

                for eval_dr in evaluations:
                    acc1, auc = accuracy_dr(outputs.clone(), targets.clone(), args, topk=(1, 1), eval_dr=eval_dr)
                    print("Evaluation: {} Acc1: {} Auc: {} ".format(eval_dr, acc1[0], auc))
                    accs += [acc1]
                    aucs += [auc]
                    save_results[eval_dr] = [acc1, auc]

                np.save(filename, save_results)
                acc5 = acc1
                '''
                # Sensitivity vs Specificity for Referrable vs Non-Referrable
                specificities = []
                sensitivities = []
                thresholds = np.arange(0,1.0,0.005)
                for th in thresholds: 
                    TP, TN, FP, FN = dr_confusion_matrix(outputs.clone(), targets.clone(), th)
                    kepsilon = 1e-7
                    specificities += [torch.div(TN, TN + FP + kepsilon)]
                    sensitivities += [torch.div(TP, TP + FN + kepsilon)]

                #print(specificities, sensitivities)
                import matplotlib.pyplot as plt
                plt.plot(specificities, sensitivities, 'o-')
                plt.ylabel('Specificity')
                plt.xlabel('Sensitivity')
                plt.xlim([0,1])
                plt.ylim([0,1])
                
                plt.savefig('/opt/code/Specifivity_Sensitivity.png', bbox_inches='tight')
                '''
            else:
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            acc1 = acc1.cpu().numpy()[0]
            acc5 = acc5.cpu().numpy()[0]
            if(bootstrap_samples>0):
                np.random.seed(42)
                bootstrap_ids = [np.random.choice(len(outputs),replace=True,size=len(outputs)) for _ in range(bootstrap_samples)]
                acc1_diff = []
                acc5_diff = []
                for b in bootstrap_ids:
                    acc1tmp, acc5tmp = accuracy(outputs[b], targets[b], topk=(1, 5))
                    acc1_diff.append(acc1tmp.cpu().numpy()[0]-acc1)
                    acc5_diff.append(acc5tmp.cpu().numpy()[0]-acc5)
                acc1_low = acc1 + np.percentile(acc1_diff, ((1.0-bootstrap_alpha)/2.0) * 100)
                acc5_low = acc5 + np.percentile(acc5_diff, ((1.0-bootstrap_alpha)/2.0) * 100)
                acc1_high = acc1 + np.percentile(acc1_diff, (bootstrap_alpha+((1.0-bootstrap_alpha)/2.0)) * 100)
                acc5_high = acc5 + np.percentile(acc5_diff, (bootstrap_alpha+((1.0-bootstrap_alpha)/2.0)) * 100)
                print(output_prefix+' bootstrap(95): Acc@1 (low,pt,high) ',acc1_low,acc1,acc1_high, 'Acc@5 (low,pt,high)',acc5_low,acc5,acc5_high)
            else:
                print(output_prefix+': Acc@1 ',acc1,"Acc@5 ",acc5)
            return acc1

            
def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.print_freq>0 and i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    print("Saving model at: {}".format(filename))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename),'model_best.pth.tar'))


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        #if 'fc.weight' in k or 'fc.bias' in k or 'classifier.weight' in k or 'classifier.bias' in k:
        #    continue
        if 'fc.1.weight' in k or 'fc.1.bias' in k or 'fc.4.weight' in k or 'fc.4.bias' in k or  'classifier.1.weight' in k or 'classifier.1.bias' in k or  'classifier.4.weight' in k or 'classifier.4.bias' in k:
            continue
        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    if args.discriminative_lrs:
        for i,param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr/pow(10,len(optimizer.param_groups)-1-i) #for discriminative lrs
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_dr(output, target, args, topk=(1,), eval_dr='quinary'):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    def rocaucscore(target, output):
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return roc_auc_score(target, output, average='micro')
    def roccurve(target, output):
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        return roc_curve(target, output)
    
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        #pdb.set_trace()
        op = nn.Softmax(dim=1)(output)
        #op = output
        if eval_dr == 'binary_rdr':
            output = torch.stack([op[:,:3].sum(1), op[:,3:].sum(1)],dim=1)
            _, pred = output.topk(maxk-3, 1, True, True)
            pred = pred.t()

            target[target==2]=0
            target[target==1]=0
            target[target==4]=1
            target[target==3]=1
            #pdb.set_trace()
            
            target = torch.zeros(len(target), 2).scatter_(1, target.unsqueeze(1), 1.)

            auc = macro_auc(output.argmax(1),target.argmax(1))
            roc_auc_curve = roccurve(target.argmax(1), output.argmax(1))
            #print("2ary AUC: {}".format(auc))
            
        elif eval_dr == 'binary_norm':
          if args.eval_criteria != 'binary_norm':
            output = torch.stack([op[:,0], op[:,1:].sum(1)],dim=1)
            _, pred = output.topk(maxk-3, 1, True, True)
            pred = pred.t()

            target[target==2]=1
            target[target==4]=1
            target[target==3]=1

            target = torch.zeros(len(target), 2).scatter_(1, target.unsqueeze(1), 1.)
            auc = macro_auc(output.argmax(1),target.argmax(1))
            roc_auc_curve = roccurve(target.argmax(1), output.argmax(1))
          
          else:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            auc = macro_auc(op[:,1:].sum(1),target)
            #roc_auc_curve = roccurve(target, output.argmax(1))
          

        elif eval_dr == 'binary_dme':
          if args.eval_criteria != 'binary_dme':
            output = torch.stack([op[:,:2].sum(1), op[:,2:].sum(1)],dim=1)
            _, pred = output.topk(maxk-3, 1, True, True)
            pred = pred.t()

            target[target==1]=0
            target[target==2]=1
            target[target==4]=1
            target[target==3]=1

            target = torch.zeros(len(target), 2).scatter_(1, target.unsqueeze(1), 1.)
            auc = macro_auc(output.argmax(1),target.argmax(1))
            roc_auc_curve = roccurve(target.argmax(1), output.argmax(1))
          else:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            auc = macro_auc(output[:,2:].sum(1),target)
            #roc_auc_curve = roccurve(target, output.argmax(1))
          
          #print("2ary AUC: {}".format(auc))
            
        elif eval_dr == 'ternary':
            output = torch.stack([op[:,0], op[:,1:3].sum(1), op[:,3:].sum(1)],dim=1)
            _, pred = output.topk(maxk-2, 1, True, True)
            pred = pred.t()
            
            target[target==2]=1
            target[target==3]=2            
            target[target==4]=2

            target = torch.zeros(len(target), 3).scatter_(1, target.unsqueeze(1), 1.)

            auc = rocaucscore(target,output)

            #print("3ary AUC: {}".format(auc))

        elif eval_dr == 'quarternary':
            # if args.eval_dataset == "messidor1":
            #     output = op #torch.stack([op[:,0], op[:,1:3].sum(1), op[:,3]],dim=1)
            # else:
            output = torch.stack([op[:,0], op[:,1:3].sum(1), op[:,3], op[:,4]],dim=1)
            _, pred = output.topk(maxk-1, 1, True, True)
            pred = pred.t()

            target[target==2]=1
            target[target==3]=2
            target[target==4]=3

            target = torch.zeros(len(target), 4).scatter_(1, target.unsqueeze(1), 1.)
            
            auc = rocaucscore(target,output)
            #print("4ary AUC: {}".format(auc))

        else:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()

            pseudo_target = target.clone()
            pseudo_target[pseudo_target>=1]=1
            auc = macro_auc(op[:,1:].sum(1),pseudo_target)
            
            #target = torch.zeros(len(target), 5).scatter_(1, target.unsqueeze(1), 1.)
            #auc = rocaucscore(target,output)

            #print("5ary AUC: {}".format(auc))

        # if eval_dr == 'quarternary':
        #     pred[pred==2]=1
        #     target[target==2]=1
        # elif eval_dr == 'ternary':
        #     pred[pred==2]=1
        #     target[target==2]=1
        #     pred[pred==4]=3
        #     target[target==4]=3
        # elif eval_dr == 'binary':
        #     #Referrable vs Non-Referrable
        #     pred[pred==2]=0
        #     target[target==2]=0
        #     pred[pred==1]=0
        #     target[target==1]=0
        #     pred[pred==4]=3
        #     target[target==4]=3
        # else:
        #     pass

        if "binary" not in args.eval_criteria:
            #correct = pred.eq(target.argmax(1).view(1, -1).expand_as(pred))
            correct = output.argmax(1)==target
        else:
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            
        res = []
        for k in topk:
            #correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            correct_k = correct.contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res[0], auc

        

def dr_confusion_matrix(pred, target, threshold, nb_classes=2):

    #pdb.set_trace()
    #pred_non_ref = pred[]

    pred1 = pred[:,:3].sum(1)
    pred2 = pred[:,3:].sum(1)

    pred_new = torch.stack([pred1, pred2], dim=1) 
    
    pred = pred_new[:,1] > threshold
    pred = pred.int()
    nb_samples = pred.shape[0]

    target[target==2]=0
    target[target==1]=0
    target[target==4]=3
    
    target[target==3]=1

    target = target == 1 
    target = target.int()
    
    #pred = torch.argmax(output, 1)
    #target = torch.randint(0, nb_classes, (nb_samples,))
    
    conf_matrix = torch.zeros(nb_classes, nb_classes)
    for t, p in zip(target, pred):
        #print(t,p)
        conf_matrix[t, p] += 1

        #print('Confusion matrix\n', conf_matrix)
        
    TP = conf_matrix.diag()
    for c in range(nb_classes):
        idx = torch.ones(nb_classes).byte()
        idx[c] = 0
        # all non-class samples classified as non-class
        TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
        # all non-class samples classified as class
        FP = conf_matrix[idx, c].sum()
        # all class samples not classified as class
        FN = conf_matrix[c, idx].sum()
            
        #print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
        #    c, TP[c], TN, FP, FN))
    return TP[c], TN, FP, FN

def macro_auc(output, target):
   with torch.no_grad():
        output_np= output.detach().cpu().numpy()
        target_np= target.cpu().numpy()
        target_labels = np.sum(target_np,axis=0)>0
        if(np.all(target_labels)):
            return roc_auc_score(target_np,output_np)
        else:
            return 0#macro auc undefined on this batch

def label_aucs(output, target):
   with torch.no_grad():
        output_np= output.detach().cpu().numpy()
        target_np= target.cpu().numpy()
        return np.array(roc_auc_score(target_np,output_np,average=None))
            
if __name__ == '__main__':
    main()
