# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

   
def bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers
        
class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap,self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

            
def create_head1d(nf, nc, lin_ftrs=None, ps=0.5, bn_final:bool=False, bn:bool=True, act="relu", concat_pooling=True):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes; added bn and act here"
    lin_ftrs = [2*nf if concat_pooling else nf, nc] if lin_ftrs is None else [2*nf if concat_pooling else nf] + lin_ftrs + [nc] #was [nf, 512,nc]
    ps = [ps] if not(isinstance(ps,list)) else ps
    if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True) if act=="relu" else nn.ELU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = []
    for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
        layers += bn_drop_lin(ni,no,bn,p,actn)
    if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)
            
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()
        
class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, mlp2=False, n_classes_ss=None, fc_ss=False, scl=False, pseudo_labels=False,pseudo_labels_threshold=0.95,label_dim=0,reduced_stem=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        mlp: mlp head a la SimCLR
        mlp2: mlp head with additional layer al la SimCLR2
        n_classes_ss: number of classes for semi-supervised
        fc_ss: fully connected output head for semi-supervised
        scl: supervised contrastive loss (incorporating label information- if available)
        pseudo_labels: use pseudo-labels (if fc_ss enabled and scl enabled)
        pseudo_labels_threhold: probability value above which the value is used a pseudo-label (a la Fix match)
        label_dim: 0 for single label
        reduced_stem: reduce stem of the encoders as done by SimCLR (for small datasets such as Cifar)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if(reduced_stem):#hack- simply replace layers - only works for resnet architectures
            self.encoder_q.conv1=nn.Conv2d(3,self.encoder_q.conv1.out_channels,kernel_size = (3,3),stride=(1,1),padding=(1,1),bias=False)
            self.encoder_q.max_pool = nn.Identity()
            self.encoder_k.conv1=nn.Conv2d(3,self.encoder_k.conv1.out_channels,kernel_size = (3,3),stride=(1,1),padding=(1,1),bias=False)
            self.encoder_k.max_pool = nn.Identity()
            
        #semi-supervised head
        self.fc_ss = fc_ss
        self.pseudo_labels = pseudo_labels
        self.pseudo_labels_threshold = pseudo_labels_threshold
        self.label_dim = label_dim
        
        if(fc_ss is True and n_classes_ss is not None):
            if(hasattr(self.encoder_q,"fc")):#resnet
                dim_mlp = self.encoder_q.fc.weight.shape[1]
            else:#densenet
                dim_mlp = self.encoder_q.classifier.weight.shape[1]
            #find GlobalAveragePooling to attach output hook (shifted between reduced/non-reduced due to nn.Identity)
            adaptive_idx = np.where([isinstance(x,nn.AdaptiveAvgPool2d) for x in self.encoder_q.children()])[0][0]
            self.topless_hook = Hook(list(self.encoder_q.children())[adaptive_idx])
            self.fc_head = nn.Sequential(nn.Linear(dim_mlp,n_classes_ss))

        assert(mlp is False or mlp2 is False)    
        if mlp:  # hack: brute-force replacement
            if(hasattr(self.encoder_q,"fc")):#resnet
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
            else:#densenet
                dim_mlp = self.encoder_q.classifier.weight.shape[1]
                self.encoder_q.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.classifier)
                self.encoder_k.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.classifier)
        elif mlp2:
            if(hasattr(self.encoder_q,"fc")):#resnets
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
            else:
                dim_mlp = self.encoder_q.classifier.weight.shape[1]
                self.encoder_q.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.classifier)
                self.encoder_k.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(),nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.classifier)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        #supervised contrastive loss
        self.scl = scl
        if(scl):
            if(self.label_dim ==0): #single-label
                self.register_buffer("queue_lbl", torch.zeros(K, dtype=torch.long))
            else: #multi-label
                self.register_buffer("queue_lbl", torch.zeros((K, self.label_dim), dtype=torch.long))
            self.register_buffer("queue_lbl_set", torch.zeros(K, dtype=torch.long))#none of them active

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, lbl=None, lbl_set=None):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        if(self.scl):#update also labels
            lbl = concat_all_gather(lbl)
            lbl_set = concat_all_gather(lbl_set)      
                
            self.queue_lbl[ptr:ptr + batch_size] = lbl
            self.queue_lbl_set[ptr:ptr + batch_size] = lbl_set
            
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, im_r=None, lbl=None, lbl_set=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images

            im_r: a batch of mildly augmented images (for semi-supervised)
            
            lbl: labels (for semi-supervised)
            lbl_set: 0 for the labels to discard 1 for the labels to use (for semi-supervised)
        Output:
            logits, targets (, logits, targets)
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        #semi-supervised head
        if(self.fc_ss):
            #qr = self.encoder_q(torch.cat([im_q,im_r],dim=0))#concatenate the two inputs for a single encoder pass- pytorch will also handle two forward passes correctly
            #q = qr[:im_q.size(0)]
            #logits_fc = self.fc_head(torch.flatten(self.topless_hook.output[im_q.size(0):],1))
            self.encoder_q(im_r)#just the forward pass to grab the intermediate output via hook
            logits_fc = self.fc_head(torch.flatten(self.topless_hook.output,1))
            logits_ss = logits_fc[lbl_set==1]
            labels_ss = lbl[lbl_set==1]
            
            if(self.pseudo_labels):
                if(self.label_dim==0):#single-label
                    probs_fc, args_fc = torch.max(nn.functional.softmax(logits_fc,dim=-1),dim=1) #single-label for now
                    cond1 = (probs_fc>self.pseudo_labels_threshold)*(lbl_set==0)
                    lbl[cond1] = args_fc[cond1]
                else:
                    preds_fc = torch.sigmoid(logits_fc)>self.pseudo_label_threshold*(lbl_set==0).unsqueeze(1)
                    cond1 = torch.sum(preds_fc,dim=1) > 0 #at least one confident prediction
                    lbl[cond1] = preds_fc[cond1]
                lbl_set[cond1] = 2 #set to specific value to distinguish from real labels
            
        q = nn.functional.normalize(q, dim=1)
   
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        #print(q.shape, k.shape, self.queue.shape)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        if(self.scl): #supervised contrastive loss
            if(self.label_dim==0):#single label
                cond1 = (lbl.unsqueeze(1) == self.queue_lbl.clone().detach()) #NxK
            else:
                cond1 = torch.sum(lbl.unsqueeze(2) == self.queue_lbl.clone().detach(),dim=1)==self.label_dim #NxK all labels have to match exactly to count as positive
            cond2 = (lbl_set.unsqueeze(1)==1)*(self.queue_lbl_set.clone().detach()>=1) #NxK (>=1 includes real and pseudo-labels)
            cond0 = torch.ones(logits.shape[0],1).to(logits.device) #Nx1: copy from own batch always positive (as for standard contrastive loss)
            labels = torch.cat([cond0,(cond1*cond2).float()],dim=1) #Nx(1+K)
        else: #standard contrastive loss
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k,lbl if self.scl else None, lbl_set if self.scl else None)

        if(self.fc_ss):
            return logits, labels, logits_ss, labels_ss
        else:
            return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


# still required?
class ResNetSimCLR(nn.Module):

    def __init__(self, base_model="resnet18", out_dim=64):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            return self.resnet_dict[model_name]
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet_CIFAR10(nn.Module):
    
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, out_dim=128):
        super(ResNet_CIFAR10, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        '''
        num_ftrs = 512 * block.expansion
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)
        #'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = x.squeeze()
        
        # h = x
        # x = self.l1(x)
        # x = self.relu(x)
        # x = self.l2(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet_CIFAR10(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class MyResnetHybrid(nn.Module):
    def __init__(self, base_resnet, vae):
        super(MyResnetHybrid, self).__init__()#, vae)
        
        encoder = list(base_resnet.children())[:-1]
        self.base_resnet = nn.Sequential(*encoder)

        #avgpool = list(base_resnet.children())[-2]
        
        #self.avgpool = avgpool
        self.fc = list(base_resnet.children())[-1:][0]
        self.vae = vae 
        
    def forward(self, x):
        #import pdb; pdb.set_trace()
        h = self.base_resnet(x)
        #h = torch.flatten(self.avgpool(h),1)
        h = torch.flatten(h,1)
        h = h/h.max()
        recon_h, mu, logvar = self.vae(h)
        recon_h.requires_grad = False
        f = self.fc(recon_h)
        f = torch.sigmoid(f)
        return f, [recon_h, h, mu, logvar]

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fcb1 = nn.Linear(512, 512)
        self.fcb2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 2048)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fcb1(h2), self.fcb2(h2)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.nn.Sigmoid()(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 2048))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
