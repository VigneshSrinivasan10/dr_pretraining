###############
#generic
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.nn.functional as F

import torchvision
import os
import argparse
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint#, LearningRateMonitor
import copy

#################
#specific
import torchvision.models as models

from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np

from moco.dataloader import *
from moco.builder import AdaptiveConcatPool2d, create_head1d
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import math
from cluster_logging import export_results

from medical.pesg_auc import PESG_AUC

from itertools import chain
import re

####################################################################################################
# AUX. FUNCTIONS
####################################################################################################

def init_weights(m):
    if type(m)== nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
            
def get_paramgroups(model,hparams,resnet=True):
    #https://discuss.pytorch.org/t/implementing-differential-learning-rate-by-parameter-groups/32903
    if(resnet):
        pgs = [[model.conv1,model.bn1,model.layer1,model.layer2], [model.layer3,model.layer4],[model.fc]]
    else:#densenet
        pgs = [[model.features],[model.classifier]]
    pgs = [[p.parameters() for p in pg] for pg in pgs]
    lgs = [{"params":chain(*pg), "lr":hparams.lr*pow(hparams.discriminative_lr_factor,len(pgs)-1-i)} for i,pg in enumerate(pgs)]
    return lgs


def _freeze_bn_stats(model, freeze=True):
    for m in model.modules():
        if(isinstance(m,nn.BatchNorm1d) or isinstance(m,nn.BatchNorm2d)):
            if(freeze):
                m.eval()
            else:
                m.train()
                
def sanity_check(model, state_dict_pre):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading state dict for sanity check")
    state_dict = model.state_dict()

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.1.weight' in k or 'fc.1.bias' in k or 'classifier.1.weight' in k or 'classifier.1.bias' in k:
            continue


        assert ((state_dict[k].cpu() == state_dict_pre[k].cpu()).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")
    
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

####################################################################################################
# MAIN MODULE
####################################################################################################
        
class CXRLightning(pl.LightningModule):

    def __init__(self, hparams):
        super(CXRLightning, self).__init__()
        
        self.hparams = hparams
        self.lr = self.hparams.lr
        
        if hparams.dataset.startswith("chexpert14") or hparams.dataset.startswith("chexphoto14") or hparams.dataset.startswith("mimic_cxr") or hparams.dataset.startswith("cxr14"):
            self.num_classes = 14
        elif hparams.dataset.startswith("chexpert") or hparams.dataset.startswith("chexphoto"):
            self.num_classes = 5
        
        if(self.hparams.auc_maximization):
            #prior class probabilities are just dummy values here
            self.criterion = auc_loss([1./self.num_classes]*self.num_classes)#num_classes
        else:
            self.criterion = F.binary_cross_entropy_with_logits #F.cross_entropy

        self.model = models.__dict__[hparams.arch](num_classes=self.num_classes)
        self.is_resnet = hasattr(self.model,"fc") #distinguish resnet vs densenet

        self.hparams.lin_ftrs_head=eval(self.hparams.lin_ftrs_head)
        if(self.hparams.linear_eval and len(self.hparams.lin_ftrs_head)>0):#overrides lin_ftrs
            print("Linear evaluation: overriding lin-ftrs-head argument")
            self.hparams.lin_ftrs_head=[]

        if(self.hparams.pretrained == "" and self.hparams.discriminative_lr_factor != 1.0):
            print("Training from scratch: overriding discriminative-lr-factor argument")
            self.hparams.discriminative_lr_factor = 1.0       
            
        if(self.is_resnet):#resnet
            if(not(self.hparams.no_concat_pooling)):
                self.model.avgpool = AdaptiveConcatPool2d()
            self.model.fc = create_head1d(2048,self.num_classes,lin_ftrs=self.hparams.lin_ftrs_head,ps=self.hparams.dropout_head,concat_pooling=not(self.hparams.no_concat_pooling),bn=not(self.hparams.no_bn_head))
        else:#densenet
            assert(self.hparams.no_concat_pooling)#todo- fix this
            self.model.classifier = create_head1d(1024,self.num_classes,lin_ftrs=self.hparams.lin_ftrs_head,ps=self.hparams.dropout_head,concat_pooling=False,bn=not(self.hparams.no_bn_head))
        
    def forward(self, x):
        return self.model(x)
        
    def _step(self,data_batch, batch_idx, train):
        if(train and self.hparams.mixup_alpha>0):
             images, target_a, target_b, lam = mixup_data(data_batch[0], data_batch[1], alpha=self.hparams.mixup_alpha)
             output = model(images)
             loss = mixup_criterion(self.criterion, output, target_a, target_b, lam)
        else:
            preds = self.forward(data_batch[0])
            loss = self.criterion(preds,data_batch[1])    
        
        self.log("train_loss" if train else "val_loss", loss)
        if(train):
            return loss
        else:
            return {'loss':loss, "preds":preds.detach(), "targs": data_batch[1]}
        
    def training_step(self, train_batch, batch_idx):
        if(self.hparams.linear_eval):
            _freeze_bn_stats(self)
        return self._step(train_batch,batch_idx,True)
        
    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        return self._step(val_batch,batch_idx,False)
        
    def validation_epoch_end(self, outputs_all):
        if(self.hparams.auc_maximization):
            print("a:",self.criterion.a.mean(),"b:",self.criterion.b.mean(),"alpha:",self.criterion.alpha.mean())
        for dataloader_idx,outputs in enumerate(outputs_all): #multiple val dataloaders
            preds_all = torch.cat([x['preds'] for x in outputs])
            targs_all = torch.cat([x['targs'] for x in outputs])
            #if(single_label):
            #    preds_all = F.softmax(preds_all,dim=-1)
            #    targs_all = torch.eye(len(self.lbl_itos))[targs_all].to(preds.device) 
            #else:
            preds_all = torch.sigmoid(preds_all)
            preds_all = preds_all.cpu().numpy()
            targs_all = targs_all.cpu().numpy()
            auc = roc_auc_score(targs_all,preds_all)
            aucs = np.array(roc_auc_score(targs_all,preds_all,average=None))
            self.log("macro_auc"+str(dataloader_idx),auc)
            #self.log("label_aucs"+str(dataloader_idx),aucs)
            print("epoch",self.current_epoch,"macro_auc"+str(dataloader_idx)+":",auc)
            #label aucs
            #print("epoch",self.current_epoch,"label_auc"+str(dataloader_idx)+":",aucs)
            
    def on_fit_start(self):
        if(self.hparams.linear_eval):
            print("copying state dict before training for sanity check after training")   
            self.state_dict_pre = copy.deepcopy(self.state_dict().copy())

    
    def on_fit_end(self):
        if(self.hparams.linear_eval):
            sanity_check(self,self.state_dict_pre)

    def _prepare_df(self):
        if(self.hparams.dataset.startswith("chexpert") or self.hparams.dataset.startswith("chexphoto")):
            df_train, df_test, label_itos = prepare_chexpert_df(self.hparams.data,label_itos=label_itos_chexpert14 if (self.hparams.dataset=="chexpert14" or self.hparams.dataset =="chexphoto14") else label_itos_chexpert5,chexphoto=self.hparams.dataset.startswith("chexphoto"),full_dataset_as_test=self.hparams.dataset.endswith("eval"))
            df_val = None
        elif(self.hparams.dataset.startswith("mimic_cxr")):
            df_train, df_val, df_test, label_itos = prepare_mimic_cxr_df(self.hparams.data,use_chexpert_labels=not(self.hparams.dataset.endswith("cxr14")))
            self.hparams.add_validation_set = True
        elif(self.hparams.dataset.startswith("cxr14")):
            df_train, df_test, label_itos = prepare_cxr14_df(self.hparams.data)
            df_val = None
      
        #splits
        if(self.hparams.custom_split or self.hparams.random_split):
            df_all = pd.concat([df_train,df_val,df_test] if df_val is not None else [df_train,df_test])
            split_stratified_wrapper(df_all,fractions=[0.8,0.1,0.1],dataset=self.hparams.dataset,save_path=os.path.join(self.hparams.code_path,"moco_official/dataset_splits/"),col_subset="custom_split",filename_postfix="custom" if self.hparams.custom_split else "random2",disregard_patients=False,disregard_labels=self.hparams.random_split)
            df_all = split_stratified_wrapper(df_all,fractions=[0.8,0.1,0.1],dataset=self.hparams.dataset,save_path=os.path.join(self.hparams.code_path,"moco_official/dataset_splits/"),col_subset="custom_split",filename_postfix="custom" if self.hparams.custom_split else "random2",disregard_patients=False,disregard_labels=self.hparams.random_split)#load split with all ranks
            df_train = df_all[df_all.custom_split==0].copy()
            df_val = df_all[df_all.custom_split==1].copy()
            df_test = df_all[df_all.custom_split==2].copy()
            self.hparams.add_validation_set = True

        elif(self.hparams.add_validation_set and not(self.hparams.dataset.startswith("mimic_cxr"))): #split off validation set if desired (for consistency with downstream training)
            split_stratified_wrapper(df_train,fractions=[0.9,0.1],dataset=self.hparams.dataset,save_path=os.path.join(self.hparams.code_path,"moco_official/dataset_splits/"),col_subset="val",filename_postfix="val")
            df_train = split_stratified_wrapper(df_train,fractions=[0.9,0.1],dataset=self.hparams.dataset,save_path=os.path.join(self.hparams.code_path,"moco_official/dataset_splits/"),col_subset="val",filename_postfix="val")#load split with all ranks
            df_train_new = df_train[df_train.val==0].copy()
            df_val = df_train[df_train.val==1].copy()
            df_train = df_train_new

        if(self.hparams.training_fraction<1.0): #reduced training set
            filename_postfix = "custom_" if self.hparams.custom_split else ("wo_val_" if self.hparams.add_validation_set else "")
            filename_postfix +="subset_"+str(self.hparams.training_fraction)

            split_stratified_wrapper(df_train,fractions=[1-self.hparams.training_fraction,self.hparams.training_fraction],dataset=self.hparams.dataset,save_path=os.path.join(self.hparams.code_path,"moco_official/dataset_splits/"),col_subset="subset",filename_postfix=filename_postfix)
            df_train = split_stratified_wrapper(df_train,fractions=[1-self.hparams.training_fraction,self.hparams.training_fraction],dataset=self.hparams.dataset,save_path=os.path.join(self.hparams.code_path,"moco_official/dataset_splits/"),col_subset="subset",filename_postfix=filename_postfix)
            df_train = df_train[df_train.subset==1].copy() #remove the unlabelled samples for now (no semi-supervised training)
        return df_train, df_val, df_test, label_itos

    def _prepare_transformations(self):
        if(self.hparams.imagenet_stats):
            normalize = transforms.Normalize(mean=imagenet_stats[0], std=imagenet_stats[1])
        else:
            normalize = transforms.Normalize(mean=chexpert_stats[0], std=chexpert_stats[1])

        #these are the "Comparison of Deep Learning Approaches... transformations
        train_transforms = transforms.Compose([
                transforms.RandomRotation(10 if self.hparams.dataset.startswith("cxr14") else 15),#was 7 in the original reference- these are the values from CheXclusion
                transforms.RandomResizedCrop(self.hparams.image_size),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        test_transforms = transforms.Compose([
            transforms.Resize(int(1.14286*self.hparams.image_size)), # was 256
            transforms.CenterCrop(self.hparams.image_size), # was 224
            transforms.ToTensor(),
            normalize,
        ])
        return train_transforms, test_transforms
       
    def prepare_data(self):
        if(self.hparams.create_splits):
            self._prepare_df()
    
    def setup(self, stage):
        train_transforms, test_transforms = self._prepare_transformations()
        
        print("Setting up data...")
        df_train, df_val, df_test, label_itos = self._prepare_df()
        print("Done. Labels:", label_itos)
        
        if(self.hparams.auc_maximization):#assign proper prior class probabilities
            self.criterion.ps = torch.from_numpy(np.mean(np.stack(df_train.label.values),axis=0))

        self.train_dataset = ImageDataframeDataset(df_train, train_transforms)
        self.test_dataset = ImageDataframeDataset(df_test, test_transforms)
        if(self.hparams.add_validation_set or self.hparams.custom_split):
            self.val_dataset = ImageDataframeDataset(df_val, test_transforms)
        else:#dummy copy of the test set in case no proper validation set is available
            self.val_dataset = self.test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, drop_last = True)
        
    def val_dataloader(self):
        return [DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4), DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=4)]
        
    def configure_optimizers(self):
        if(hparams.auc_maximization):
            if(self.hparams.linear_eval or self.hparams.train_head_only):
                params = [{"params":(self.model.fc[-1].parameters() if len(self.hparams.lin_ftrs_head)>0  else self.model.fc.parameters()) if self.is_resnet else (self.model.classifier[-1].parameters() if len(self.hparams.lin_ftrs_head)>0 else self.model.classifier.parameters()), "lr":self.lr},{"params":iter([self.criterion.a,self.criterion.b]), "lr":100*self.lr},{"params":iter([self.criterion.alpha]), "lr":100*self.lr, "is_alpha":True}]
            else:
                params = get_paramgroups(self.model,self.hparams,resnet=self.is_resnet)+[{"params":iter([self.criterion.a,self.criterion.b]), "lr":100*self.lr},{"params":iter([self.criterion.alpha]), "lr":100*self.lr, "is_alpha":True}]
            opt = PESG_AUC
            
        else:
            if(self.hparams.optimizer == "sgd"):
                opt = torch.optim.SGD
            elif(self.hparams.optimizer == "adam"):
                opt = torch.optim.AdamW
            else:
                raise NotImplementedError("Unknown Optimizer.")
            
            if(self.hparams.linear_eval or self.hparams.train_head_only):
                params = self.model.fc.parameters() if self.is_resnet else self.model.classifier.parameters()
            elif(self.hparams.discriminative_lr_factor != 1.):#discrimative lrs
                params = get_paramgroups(self.model,self.hparams,resnet=self.is_resnet)
            else:
                params = self.parameters()
            
        optimizer = opt(params, self.lr, weight_decay=self.hparams.weight_decay)

        return optimizer
        
    def load_weights_from_checkpoint(self, checkpoint):
        """ Function that loads the weights from a given checkpoint file. 
        based on https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        if("state_dict" in checkpoint.keys()):#lightning style
            pretrained_dict = checkpoint["state_dict"]
        else:
            pretrained_dict = checkpoint

        if("fc.bias" in pretrained_dict.keys() or "classifier.bias" in pretrained_dict.keys()):#e.g. pretrained imagenet classifier
            print("Loading weights from pretrained backbone...")
            pretrained_dict_classes = pretrained_dict["fc.bias"].size()[0] if self.is_resnet else pretrained_dict["classifier.bias"].size()[0]
            if(pretrained_dict_classes != self.num_classes):
                print("number of classes in state dict (",pretrained_dict_classes,") do not match ",self.num_classes,". Discarding fc layer.")
                if(self.is_resnet):
                    del pretrained_dict["fc.bias"]
                    del pretrained_dict["fc.weight"]
                    self.model.fc.apply(init_weights)
                else:
                    del pretrained_dict["classifier.bias"]
                    del pretrained_dict["classifier.weight"]
                    self.model.classifier.apply(init_weights)
        elif("module.encoder_q.fc.weight" in pretrained_dict.keys() or "module.encoder_q.classifier.weight" in pretrained_dict.keys()):
            print("Loading weights from self-supervised pretraining...")
            for k in list(pretrained_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc') and not k.startswith('module.encoder_q.classifier'):
                    # remove prefix
                    pretrained_dict[k[len("module.encoder_q."):]] = pretrained_dict[k]
                #semi-supervised pretraining (keep the semi-supervised head)
                if(k=="module.fc_ss"):
                    if(self.is_resnet):#resnet
                        pretrained_dict["module.fc"]=pretrained_dict[k]
                    else:#densenet
                        pretrained_dict["module.classifier"]=pretrained_dict[k]
                # delete renamed or unused k
                del pretrained_dict[k]
        else:
            print("Loading weights...")
 
        #potentially correct for the fact the the model is hidden as model. inside the lightning module
        pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        
        for _ in range(len(pretrained_dict)):
            key,value = pretrained_dict.popitem(False)
            #issues with renamed densenet weights c.f. _load_state_dict in torchvision densenet
            res = pattern.match(key)
            if res:
                key = res.group(1) + res.group(2)
            pretrained_dict["model."+key if not(key.startswith("model.")) else key] = value
            
            
        model_dict = self.state_dict()
        missing_keys = [m for m in model_dict.keys() if not(m in pretrained_dict)]
        missing_keys_wo_num_batches_tracked = [m for m in missing_keys if not(m.endswith("num_batches_tracked"))]
        print("INFO:",len(model_dict)-len(missing_keys),"of",len(model_dict),"keys were matched.\n",len(missing_keys_wo_num_batches_tracked),"missing keys (disregarding *.num_batches_tracked):",missing_keys_wo_num_batches_tracked) 
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


#####################################################################################################
#ARGPARSERS
#####################################################################################################
def add_model_specific_args(parser):
    parser.add_argument('--arch',default="densenet121", type=str,help='torchvision architecture- presently only resnets and densenets')
    parser.add_argument("--dropout-head", type=float, default=0.5)
    parser.add_argument("--train-head-only", action="store_true", help="freeze everything except classification head (note: --linear-eval defaults to no hidden layer in classification head)")
    parser.add_argument('--no-bn-head', action='store_true', help="use no batch normalization in classification head")
    parser.add_argument('--no-concat-pooling', action='store_true', help="use no concat pooling and standard mean pooling instead (applies to resnets only)")
    parser.add_argument('--lin-ftrs-head', type=str, default="[512]", help='linear filters for head (as string e.g. [1024] or [] for no extra hidden layers)')

    return parser

def add_default_args():
    parser = argparse.ArgumentParser(description='PyTorch Lightning CXR Training')
    parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
    parser.add_argument("--dataset",type=str,help="chexpert/chexpert14/mimic_cxr/cxr14/chexphoto14/chexphoto", default="mimic_cxr")
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.0015, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained checkpoint (default: none)')
    parser.add_argument('--optimizer', default='adam', help='sgd/adam')#was sgd
    parser.add_argument('--output-path', default='.', type=str,dest="output_path",
                        help='output path')
    parser.add_argument('--code-path', default='/opt/submit/', type=str, dest="code_path", help='code path')                   
    parser.add_argument('--metadata', default='', type=str,
                        help='metadata for output')
    
    parser.add_argument("--gpus", type=int, default=1, help="number of gpus")
    parser.add_argument("--num-nodes", dest="num_nodes", type=int, default=1, help="number of compute nodes")
    parser.add_argument("--precision", type=int, default=32, help="16/32")
    parser.add_argument("--distributed-backend", dest="distributed_backend", type=str, default=None, help="None/ddp")
    parser.add_argument("--accumulate", type=int, default=1, help="accumulate grad batches (total-bs=accumulate-batches*bs)")
        
    parser.add_argument("--linear-eval", action="store_true", help="linear evaluation instead of full finetuning",  default=False )

    parser.add_argument('--imagenet-stats', action='store_true', help='use imagenet stats instead of dataset stats (use with pretrained imagenet models)')
    parser.add_argument('--image-size', default=224, type=int, help='image size in pixels')
    parser.add_argument('--add-validation-set', action='store_true', help='split off validation set')
    parser.add_argument('--create-splits', action='store_true', help='option to create splits in a multi-process environment')
    parser.add_argument('--custom-split', action='store_true', help='custom stratified split 80:10:10')
    parser.add_argument('--random-split', action='store_true', help='random split diregarding patients 80:10:10')
    parser.add_argument('--training-fraction', default=1.0, type=float,dest="training_fraction", help='fraction of training examples to be used.')
    #parser.add_argument('--bootstrap-samples', default=0, type=int, dest='bootstrap_samples', help='number of bootstrap samples during evaluation (0 for no bootstrap)')
        
    parser.add_argument('--mixup-alpha', type=float, default=0.0, help='alpha>0 use mixup during training (default choice alpha=1)')
    parser.add_argument("--discriminative-lr-factor", type=float, help="factor by which the lr decreases per layer group during finetuning", default=0.1)
    
    
    parser.add_argument("--lr-find", action="store_true",  help="run lr finder before training run", default=False )
    
    parser.add_argument("--auc-maximization", action="store_true", help="direct auc maximization",  default=False )
    parser.add_argument('--refresh-rate', default=0, type=int, help='progress bar refresh rate (0 for disabled)')
    
    return parser
             
###################################################################################################
#MAIN
###################################################################################################
# example call: python main_lincls_lightning.py /media/strodthoff/data2/cxr_data/ --code-path /home/strodthoff/work/unsupervised_learning --output-path /home/strodthoff/work/unsupervised_learning/runs/chexpert14 --pretrained /home/strodthoff/work/unsupervised_learning/runs/hub/densenet121-a639ec97.pth --refresh-rate 1 --lr 0.0015 --epochs 100 --batch-size 64 --custom-split --arch densenet121 --imagenet-stats --no-concat-pooling

if __name__ == '__main__':
    parser = add_default_args()
    parser = add_model_specific_args(parser)
    hparams = parser.parse_args()
    hparams.executable = "main_lincls_lightning"

    if not os.path.exists(hparams.output_path):
        os.makedirs(hparams.output_path)
        
    model = CXRLightning(hparams)
    
    if(hparams.pretrained!=""):
        print("Loading pretrained weights from",hparams.pretrained)
        model.load_weights_from_checkpoint(hparams.pretrained)
        if(hparams.auc_maximization):#randomize top layer weights
            if(hparams.train_head_only):
                if(self.is_resnet):
                    if(len(model.hparams.lin_ftrs_head)>0):
                        model.fc.apply(init_weights)
                    else:
                        model.fc[-1].apply(init_weights)
                else:
                    if(len(model.hparams.lin_ftrs_head)>0):
                        model.classifier[-1].apply(init_weights)
                    else:
                        model.classifier.apply(init_weights)

    logger = TensorBoardLogger(
        save_dir=hparams.output_path,
        #version="",#hparams.metadata.split(":")[0],
        name="")
    print("Output directory:",logger.log_dir)    
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename="best_model",#hparams.output_path
        save_top_k=1,
		save_last=True,
        verbose=True,
        monitor='macro_auc0',#val_loss/dataloader_idx_0
        mode='max',
        prefix='')
    #lr_monitor = LearningRateMonitor()

    trainer = pl.Trainer(
        #overfit_batches=0.01,
        auto_lr_find = hparams.lr_find,
        accumulate_grad_batches=hparams.accumulate,
        max_epochs=hparams.epochs,
        min_epochs=hparams.epochs,
        
        default_root_dir=hparams.output_path,
        
        num_sanity_val_steps=0,
        
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        callbacks = [],#lr_monitor],
        benchmark=True,
    
        gpus=hparams.gpus,
        num_nodes=hparams.num_nodes,
        precision=hparams.precision,
        distributed_backend=hparams.distributed_backend,
        
        progress_bar_refresh_rate=hparams.refresh_rate,
        weights_summary='top',
        resume_from_checkpoint= None if hparams.resume=="" else hparams.resume)
        
    if(hparams.lr_find):#lr find
        trainer.tune(model)
        
    trainer.fit(model)
    

