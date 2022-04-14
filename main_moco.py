#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

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

import moco.loader
import moco.builder

import pandas as pd
import numpy as np
from pathlib import Path
from moco.dataloader import *
from cluster_logging import export_results

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
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
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

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head a la SimCLR')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')



##############################
#extra arguments introduced by us
parser.add_argument('--mlp2', action='store_true',
                    help='use additional layer in mlp head a la SimCLRv2')
parser.add_argument('--dataset', default='imagenet', 
                    help='imagenet/chexpert(14)/mimic_cxr/cxr_combined/diabetic_retinopathy/cifar10/stl10(_x)')
parser.add_argument('--aug-cx', dest="aug_cx", action='store_true',
                    help='use chexpert data augmentation')                    
parser.add_argument('--aug-dr', dest="aug_dr", action='store_true',
                    help='use diabetic retinopathy data augmentation')
parser.add_argument('--aug-jpeg', dest="aug_jpeg", action='store_true',
                    help='use jpeg instead of Gaussian Blur in the standard SimCLR transformations')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--find-stats', action='store_true',
                    help='determine dataset stats',dest="find_stats")
parser.add_argument('--output-path', default='.', type=str, dest="output_path",
                    help='output path')
parser.add_argument('--code-path', default='/opt/submit/', type=str, dest="code_path",
                    help='code path')
parser.add_argument('--save-freq', default=1, type=int,
                    metavar='N', help='save frequency (default: 1- to allow reinitialization)', dest="save_freq")
parser.add_argument('--metadata', default='', type=str,
                    help='metadata for output')
parser.add_argument('--imagenet-stats', action='store_true', dest='imagenet_stats',
                    help='use imagenet stats instead of dataset stats')
parser.add_argument('--image-size', default=224, type=int, dest='image_size',
                    help='image size in pixels')
parser.add_argument('--add-validation-set', action='store_true', dest='add_validation_set',
                    help='split off validation set')
parser.add_argument('--custom-split', action='store_true', dest='custom_split',
                    help='custom stratified split 80:10:10')
#semi-supervised stuff                    
parser.add_argument('--fc-ss', dest='fc_ss', action='store_true',
                    help='additional output head (for semi-supervised training)')                                         
parser.add_argument('--lambda-loss', default=1e-2, type=float,dest="lambda_loss",
                    help='relative factor between the two loss components (semi-supervised if fc-ss is active).')
parser.add_argument('--training-fraction', default=None, type=float,dest="training_fraction",
                    help='fraction of training examples to be used (value 1.0 allows to use supervised-contrastive-loss for the full dataset).')
parser.add_argument('--supervised-contrastive-loss', dest='supervised_contrastive_loss', action='store_true',
                    help='supervised contrastive loss (using labels where available) instead of standard contrastive loss')
parser.add_argument('--pseudo-labels', action='store_true',
                    help='use pseudo-labels (requires fc-ss and supervised-contrastive-loss)') 

def main():
    args = parser.parse_args()
    args.executable = "main_moco"
    args.semi_supervised = (args.training_fraction is not None)
    
    print(args)
    
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
        #export results


def main_worker(gpu, ngpus_per_node, args):
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
    
    #num_classes and losses for semi-supervised (reduction="sum" to avoid issues with empty batches)
    if args.dataset == "imagenet":
        num_classes = 1000
        criterion_ss = nn.CrossEntropyLoss(reduction="sum").cuda(args.gpu)
    elif args.dataset.startswith("chexpert") or args.dataset=="mimic_cxr" or args.dataset == "cxr_combined":
        num_classes = 14 if (args.dataset=="chexpert14" or args.dataset=="mimic_cxr" or args.dataset == "cxr_combined") else 5
        criterion_ss = nn.BCEWithLogitsLoss(reduction="sum").cuda(args.gpu)
    elif args.dataset == "diabetic_retinopathy":
        num_classes = 5
        criterion_ss = nn.CrossEntropyLoss(reduction="sum").cuda(args.gpu)    
    elif args.dataset == 'cifar10' or args.dataset.startswith("stl10"):
        num_classes = 10
        criterion_ss = nn.CrossEntropyLoss(reduction="sum").cuda(args.gpu)  
   
    #if args.arch == 'resnet_cifar': #special cifar resnet
    #    print("=> creating model resnet_cifar")
    #    model = moco.builder.MoCo(
    #        moco.builder.ResNet_CIFAR10,
    #        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, mlp=args.mlp, mlp2=args.mlp2, n_classes_ss=num_classes,fc_ss=args.fc_ss,scl=args.supervised_contrastive_loss,pseudo_labels=args.pseudo_labels,label_dim = num_classes if args.dataset=="chexpert" else 0)
    #else:#standard torchvision models
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, mlp=args.mlp, mlp2=args.mlp2, 
                n_classes_ss=num_classes,fc_ss=args.fc_ss,scl=args.supervised_contrastive_loss,
                pseudo_labels=args.pseudo_labels, label_dim = num_classes if (args.dataset.startswith("chexpert") or args.dataset =="mimic_cxr") else 0,
                reduced_stem=(args.dataset=="cifar10"))
        
    #print(model)

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
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    if(args.supervised_contrastive_loss):
        criterion = supervised_contrastive_loss#nn.BCEWithLogitsLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

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
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if(args.dataset == "imagenet" or args.imagenet_stats or args.dataset.startswith("stl10")):
        normalize = transforms.Normalize(mean=imagenet_stats[0],
                                     std=imagenet_stats[1])
    elif(args.dataset.startswith("chexpert") or args.dataset == "mimic_cxr" or args.dataset == "cxr_combined"):
        normalize = transforms.Normalize(mean=chexpert_stats[0],
                                     std=chexpert_stats[1])
    elif(args.dataset == "diabetic_retinopathy"):
        normalize = transforms.Normalize(mean=dr_stats[0],
                                     std=dr_stats[1])
    elif(args.dataset == "cifar10"):
        normalize = transforms.Normalize(mean=cifar_stats[0],
                                     std=cifar_stats[1])
    else:
        assert(False)

    #mild augmentation for semi-supervised
    augmentation_mild = [
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]

    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        if args.dataset != "cifar10":
            augmentation = [
                transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([moco.loader.JPEGCompress(quality=15) if args.aug_jpeg else moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        else:#cifar10
        # get a set of data augmentation transformations as described in the SimCLR paper.
        
            augmentation = [
                transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
                transforms.RandomApply([transforms.ColorJitter(0.5, 0.5, 0.5, 0.2)
                ], p=0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    if args.aug_cx:
        # similar to moco v2 but with more random noise transformations
        augmentation = [
            transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            #transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.JPEGCompress(quality=15) if args.aug_jpeg else moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            
            transforms.RandomApply([transforms.RandomChoice([
            moco.loader.RandomNoiseGrayscale(mode="gaussian",var=0.02),
            moco.loader.RandomNoiseGrayscale(mode="poisson"),
            #RandomNoiseGrayscale(mode="s&p"),
            moco.loader.RandomNoiseGrayscale(mode="speckle",var=0.02)])], p=0.5),
            transforms.ToTensor(),
            normalize
            ]

    if args.aug_dr:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            #transforms.Resize(512)
            transforms.RandomResizedCrop(args.image_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            #transforms.RandomApply([moco.loader.JPEGCompress(quality=15) if args.aug_jpeg else moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    if args.find_stats:
        augmentation = [
            transforms.ToTensor(),
        ]
    print("Constructing dataset...")
    if args.dataset == "imagenet" or args.dataset== "cifar10":
        assert(args.dataset == "imagenet" or args.image_size <= 32) #check cifar image size
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val' if args.dataset == "imagenet" else "test")
        df_train, label_itos = prepare_imagefolder_df(traindir)
        df_valid, _ = prepare_imagefolder_df(valdir,label_itos)
        #Original version: via datasets.ImageFolder
        #train_dataset = datasets.ImageFolder(
        #    traindir,
        #    moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))    
    elif args.dataset.startswith("chexpert"):
        df_train, df_valid, label_itos = prepare_chexpert_df(args.data, label_itos=label_itos_chexpert14 if args.dataset=="chexpert14" else label_itos_chexpert5)
        #could also use label_raw and a random target_transform in the dataset constructor as in 1911.06475
        #df_train["label"]=df_train["label_raw"]        
        #df_valid["label"]=df_valid["label_raw"]
        #target_transform = chexpert_label_raw_to_label_uoneslsr #chexpert_label_raw_to_label_uzeroslsr
    elif args.dataset == "mimic_cxr":
        df_train, df_valid, df_test, label_itos = prepare_mimic_cxr_df(args.data)
        df_train = pd.concat([df_train,df_valid])
        df_valid = df_test
    elif(args.dataset == "cxr_combined"):
        df_train1, df_valid1, _ = prepare_chexpert_df(args.data)
        df_train1["label"] = 0 #discard all labels
        df_valid1["label"] = 0
        df_train2, df_valid2, df_test2, _ = prepare_mimic_cxr_df(args.data)
        df_train2["label"] = 0
        df_valid2["label"] = 0
        df_test2["label"] = 0
        df_train3, df_valid3, _ = prepare_cxr14_df(args.data)
        df_train3["label"] = 0
        df_valid3["label"] = 0
        #no validation set
        df_train = pd.concat([df_train1[["image_id","patient_id","path","label"]],df_valid1[["image_id","patient_id","path","label"]],df_train2[["image_id","patient_id","path","label"]],df_valid2[["image_id","patient_id","path","label"]],df_test2[["image_id","patient_id","path","label"]],df_train3[["image_id","patient_id","path","label"]],df_valid3[["image_id","patient_id","path","label"]]])
        df_valid = df_test2[["image_id","patient_id","path","label"]]
         
    elif args.dataset == "diabetic_retinopathy":
        df_train, df_valid, label_itos = prepare_diabetic_retinopathy_df(args.data)

    if(args.custom_split):
        df_all = pd.concat([df_train,df_valid])
        if(args.rank==0):#only create splits for rank0
            split_stratified_wrapper(df_all,fractions=[0.8,0.1,0.1],dataset=args.dataset,save_path=os.path.join(args.code_path,"moco_official/dataset_splits/"),col_subset="custom_split",filename_postfix="custom")
        dist.barrier()#wait until rank0 is finished
        df_all = split_stratified_wrapper(df_all,fractions=[0.8,0.1,0.1],dataset=args.dataset,save_path=os.path.join(args.code_path,"moco_official/dataset_splits/"),col_subset="custom_split",filename_postfix="custom")#load split with all ranks
        df_train = df_all[df_all.custom_split<2].copy()
        df_valid = df_all[df_all.custom_split==2].copy()
    elif(args.add_validation_set and args.dataset !="mimic_cxr"): #split off validation set if desired (for consistency with downstream training)
        if(args.rank == 0):
            split_stratified_wrapper(df_train,fraction=[0.9,0.1],dataset=args.dataset,save_path=os.path.join(args.code_path,"moco_official/dataset_splits/"),col_subset="val",filename_postfix="val")
        dist.barrier()#wait for rank 0 to finish
        df_train = split_stratified_wrapper(df_train,fraction=[0.9,0.1],dataset=args.dataset,save_path=os.path.join(args.code_path,"moco_official/dataset_splits/"),col_subset="val",filename_postfix="val") #now load the split for all ranks
        df_train_new = df_train[df_train.val==0].copy()
        df_valid = df_train[df_train.val==1].copy()
        df_train = df_train_new

    if(args.semi_supervised and not(args.dataset.startswith("stl10"))): #semi-supervised
        if(args.rank == 0):#create split only on rank 0
            split_stratified_wrapper(df_train,fraction=[1-args.training_fraction,args.training_fraction],dataset=args.dataset,save_path=os.path.join(args.code_path,"moco_official/dataset_splits/"),col_subset="subset",filename_postfix=("custom_" if args.custom_split else "")+("wo_val_" if args.add_validation_set else "")+"subset_"+str(args.training_fraction))
        dist.barrier()#wait for rank 0 to finish
        #load the split for all ranks
        df_train = split_stratified_wrapper(df_train,fraction=[1-args.training_fraction,args.training_fraction],dataset=args.dataset,save_path=os.path.join(args.code_path,"moco_official/dataset_splits/"),col_subset="subset",filename_postfix=("custom_" if args.custom_split else "")+("wo_val_" if args.add_validation_set else "")+"subset_"+str(args.training_fraction))
    
    print("Constructing dataloader...")
    if(args.dataset.startswith("stl10")):#special STL-10 dataset
        assert(args.image_size<=96)
        fold = args.dataset.split("_")
        fold = int(fold[1]) if(len(fold)==2) else None
        label_itos = label_itos_stl10
        
        #pass --training-fraction 1.0 for semi-supervised
        assert(args.semi_supervised is False or (args.fc_ss or args.supervised_contrastive_loss))
        
        train_dataset = datasets.STL10(args.data, 
            split='train+unlabeled', 
            folds=fold, 
            transform=moco.loader.TwoCropsTransform(transforms.Compose(augmentation)) if not(args.fc_ss) else moco.loader.ThreeCropsTransform(transforms.Compose(augmentation),transforms.Compose(augmentation_mild)), 
            target_transform=target_transform_stl10 if args.semi_supervised else None)
             
    else:#default: create the actual dataset
        train_dataset = ImageDataframeDataset(df_train,
            moco.loader.TwoCropsTransform(transforms.Compose(augmentation)) if not(args.fc_ss) else moco.loader.ThreeCropsTransform(transforms.Compose(augmentation),transforms.Compose(augmentation_mild)),
            col_target_set="subset" if args.semi_supervised else None)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    history_acc1, history_acc5, best_acc1=[],[],0.
    
    print("Training started...")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        if not args.find_stats: 
            # train for one epoch
            acc1, acc5 = train(train_loader, model, criterion, optimizer, epoch, args,criterion_ss=criterion_ss if args.fc_ss else None)
            #export results after every epoch (in case the job get killed)
            if(acc1>best_acc1):
                best_acc1 = acc1
            history_acc1.append(acc1)
            history_acc5.append(acc5)
            export_results([args,{"result_epoch":epoch, "result_acc1":acc1, "result_best_acc1":best_acc1, "result_acc5":acc5, "result_history_acc1": history_acc1, "result_history_acc5": history_acc5}],os.path.join(args.output_path,"results.json"))


            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                if((epoch +1)%args.save_freq ==0 or epoch+1 == args.epochs):
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best=False, filename=os.path.join(args.output_path,'checkpoint_latest.pth.tar'))#'checkpoint_{:04d}.pth.tar'.format(epoch)
        else:
            mean, std = find_stats_dataset(train_loader)
            print('Dataset: {} | Mean: {} | Std: {}'.format(args.dataset, mean, std))
            save_results_file = 'stats_{}.txt'.format(args.dataset)
            with open(save_results_file, 'a') as the_file:
                the_file.write('Mean: {} | Std: {}'.format(mean, std))
            the_file.close()


def find_stats_dataset(loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for i, (data,_) in enumerate(loader):
        #if i%10 == 0:
        print('No samples completed: {}'.format(i))
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
        if i %10 == 0:
            print('Mean: {} | Std: {}'.format(mean, std))
            #import sys
            #sys.exit()
    mean /= nb_samples
    std /= nb_samples

    return mean, std 

def train(train_loader, model, criterion, optimizer, epoch, args, criterion_ss=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@0.2' if args.supervised_contrastive_loss else 'Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@0.5' if args.supervised_contrastive_loss else 'Acc@5', ':6.2f')
    if(args.fc_ss):
        losses_moco = AverageMeter('Loss Cont', ':.4e')
        losses_ss = AverageMeter('Loss Class', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses_moco, losses_ss, top1, top5] if args.fc_ss else [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, samples in enumerate(train_loader):
        images = samples[0]
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            if(args.fc_ss): #mildly augmented version for fc head
                images[2] = images[2].cuda(args.gpu, non_blocking=True)

        # compute output
        if(args.semi_supervised):#semi-supervised
            lbls = samples[1][0]
            lbls_set = samples[1][1]
            if args.gpu is not None:
                lbls = lbls.cuda(args.gpu, non_blocking=True)
                lbls_set = lbls_set.cuda(args.gpu, non_blocking=True)
            if(args.fc_ss): #fc head
                output, target, output_ss, target_ss = model(im_q=images[0], im_k=images[1],im_r=images[2],lbl=lbls,lbl_set=lbls_set)
                loss_moco = criterion(output, target)
                losses_moco.update(loss_moco.item(), images[0].size(0))
                loss_ss = len(target_ss)/(len(target_ss)+1e-6)/(len(target_ss)+1e-6)*criterion_ss(output_ss, target_ss)# all operation should suport empty batches for pytorch>=1.5; this line assumes reduction="sum"
                losses_ss.update(loss_ss.item(), target_ss.size(0))
                loss = loss_moco + args.lambda_loss*loss_ss
            else:
                output, target = model(im_q=images[0], im_k=images[1],lbl=lbls,lbl_set=lbls_set)
                loss = criterion(output, target)
        else:    
            output, target = model(im_q=images[0], im_k=images[1])
            loss = criterion(output, target)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        losses.update(loss.item(), images[0].size(0))
        acc1, acc5 = accuracy_threshold(output, target, thresholds=[0.2,0.5]) if args.supervised_contrastive_loss else accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    # return performance       
    return float(top1.avg.cpu().numpy()), float(top5.avg.cpu().numpy())

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename),'model_best.pth.tar'))


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
        if(n>0):# and not(np.isnan(val.cpu()))):
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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        
def accuracy_threshold(output,target, thresholds=[0.2,0.5],apply_sigmoid=True):
    """metric for multi-label (supervised contrastive loss)"""
    with torch.no_grad():
        if(apply_sigmoid):
            outputs = torch.sigmoid(output)
            
        res = []
        batch_size = target.size(0)
        features = target.size(1)
        
        for t in thresholds:
            correct = target.long().eq((outputs>t).long()).float().sum()
            res.append([correct.mul_(100.0/batch_size/features)])
        return res 
        
def supervised_contrastive_loss(output, target):
    # c.f. https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    #output N,dim scalarprods/T
    #target N,dim multi-hot targets

    logits_max, _ = torch.max(output, dim=1, keepdim=True)
    logits = output - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mask = (target==1)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = - mean_log_prob_pos
    return loss.mean()
        
if __name__ == '__main__':
    main()
