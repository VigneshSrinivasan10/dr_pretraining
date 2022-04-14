import numpy as np
import torch
import torch.nn as nn
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import os
from tqdm.autonotebook import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

colors = ["lightsteelblue", "lightcoral", "royalblue", "orangered", "darkred"]

def compute_ev(feats,num_components):
    scaler = StandardScaler().fit(feats)
    feats_std = scaler.transform(feats)
    pca = PCA(n_components=num_components)
    pca.fit(feats_std)
    eigenvalues = pca.explained_variance_
    #eigenvalues = pca.lambdas_
    return eigenvalues

def compute_cn(ev,p1=99,p2=50):
    return np.abs(np.percentile(ev, p1)) / (np.abs(np.percentile(ev, p2)))
    #return np.abs(ev.max()) / (np.abs(ev.min())+1e-25)

def extract_pca_features(model,dataloader,output_path="./pca_analysis",num_batches=10,num_components=100, eligible_layers = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Linear],device="cuda"):
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    outputs = {}
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach().cpu().numpy()
        return hook

    hooks = []
    for name, layer in model.named_modules():
        if type(layer) in eligible_layers:
            hooks.append(layer.register_forward_hook(get_activation(name)))
            outputs[name] = []

    batch_size=0
    iterator = iter(dataloader)
    num_iterations = len(dataloader) if num_batches==0 else min(num_batches,len(dataloader))
    for idx in tqdm(list(range(num_iterations))):
        input, label = next(iterator)
        batch_size = input.size(0)
        output = model(input.to(device))

        for name, layer in model.named_modules():
            if type(layer) in eligible_layers:
                outputs[name] += [activation[name]]
   
  

    evs = []
    for kk, name in tqdm(enumerate(list(outputs.keys()))):
        if kk <99:  
            evs+=[compute_ev(np.vstack(outputs[name]).reshape([num_batches*batch_size,-1]),num_components)]
    cns_99_95 = [compute_cn(ev,99,95) for ev in evs]
    cns_99_90 = [compute_cn(ev,99,90) for ev in evs]
    cns_99_50 = [compute_cn(ev,99,50) for ev in evs]
    cns_99_10 = [compute_cn(ev,99, 10) for ev in evs]
    cns_99_5 = [compute_cn(ev,99, 5) for ev in evs]
  
    cns_999_95 = [compute_cn(ev,99.9, 95) for ev in evs]
    cns_999_90 = [compute_cn(ev,99.9, 90) for ev in evs]
    cns_999_50 = [compute_cn(ev,99.9, 50) for ev in evs]
    cns_999_10 = [compute_cn(ev,99.9, 10) for ev in evs]
    cns_999_5 = [compute_cn(ev,99.9, 5) for ev in evs]
  
    final_dict = {}
    final_dict['evs'] = evs
    final_dict['cns'] = cns_99_95
  
    final_dict['cns_99_95'] = cns_99_95
    final_dict['cns_99_90'] = cns_99_90
    final_dict['cns_99_50'] = cns_99_50
    final_dict['cns_99_10'] = cns_99_10
    final_dict['cns_99_5'] = cns_99_5
  
    final_dict['cns_999_95'] = cns_999_95
    final_dict['cns_999_90'] = cns_999_90
    final_dict['cns_999_50'] = cns_999_50
    final_dict['cns_999_10'] = cns_999_10
    final_dict['cns_999_5'] = cns_999_5
  
    final_dict['layers'] = [*outputs.keys()]
  
    np.save(output_path+'/all_ev.npy', final_dict)

    for h in hooks:
        h.remove()

def plot_all_ev(feature_paths = [], labels=[], output_path = "./pca_analysis", output_file="condition_number_all_layers_{}_{}_v3.pdf", figsize=(15,7)):
    #plt.rcParams.update({'font.size': 40})

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    p1s = [999]#, 99]
    p2s = [90] #[95, 90, 50, 10, 5]
    for p1 in p1s:
        for p2 in p2s:
            #plt.figure(figsize=(20,7))
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=figsize)
    
            #for idx, (method,m) in enumerate(zip(methods_fullname, methods_)):
            for idx,(fp,lbl) in enumerate(zip(feature_paths,labels)):
                final_dict = np.load(fp+'/all_ev.npy', allow_pickle=True).item()
                
                cns = final_dict['cns']
                cns = final_dict['cns_{}_{}'.format(p1,p2)][:-2]
                names = final_dict['layers'][:-2]
                
                filtered_cns = [cn for name, cn in zip(names, cns) if 'conv' in name]

                #plt.plot(cns, 'o-', label=m, color=colors[idx])
                
                if idx <2:
                    ax[0].plot(cns, 'o-', label=lbl, color=colors[idx])#linewidth=5, markersize=10
                    ax[0].set_xticks([])
                    ax[0].set_ylim([0,80])
                    ax[0].legend()
                    #ax[0].set_ylabel('Condition Number')
            
                else:
                    ax[1].plot(cns, 'o-', label=lbl, color=colors[idx])#linewidth=5, markersize=10
                    ax[1].set_xticks([])
                    ax[1].set_xlabel(r'Layers $\longrightarrow$')
                    ax[1].set_ylim([0,80])
                    ax[1].legend()
                    #ax[1].set_ylabel('Condition Number')
            
                
            fig.text(0.08, 0.5, 'Condition Number', va='center', ha='center', rotation='vertical')#, fontsize=rcParams['axes.labelsize'])
            plt.savefig(output_path+'/'+output_file.format(p1, p2), bbox_inches='tight')
            plt.show()
            
            #plt.clf()

def plot_ev(feature_paths=[],labels=[],output_path="./pca_analysis",output_file="eigenvalues_density_conv1_v3.pdf",figsize=(10,8)):
    #plt.rcParams.update({'font.size': 40})

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    evs_conv1 = []
    evs_fc1 = []
    cns_conv1 = []
    cns_fc1 = []
    for fp in feature_paths:
        final_dict = np.load(fp+'/all_ev.npy', allow_pickle=True).item()
        evs = final_dict['evs']
        names = final_dict['layers']
        evs_conv1 += [evs[0]]

    title = ['conv1', 'fc1']
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=figsize)
    kurts_conv1 = []
    kurts_fc1 = []

    for cid, (evc, fp, lbl) in tqdm(enumerate(zip(evs_conv1, feature_paths, labels))):
        evc_samples = evc / 5e0
        idxs = np.arange(evc_samples.shape[0])
        nidxs = -idxs[1:]
        idxs = np.hstack([idxs,nidxs])

        samples = [[h]*int(i) for h,i in zip(idxs, evc_samples) ]
        nsamples = [[h]*int(i) for h,i in zip(nidxs, evc_samples[1:]) ]

        final_samples = [s for sample in samples for s in sample] +  [s for sample in nsamples for s in sample]

        if cid <2:
            idx= 0
        else:
            idx = 1
            
        pd.DataFrame(final_samples).plot(kind='density',ax=ax[0,idx],lw=5, color=colors[cid], label=lbl)#,label='Kurtosis: {:.3f} '.format(kurtosis(final_samples)))
        pd.DataFrame(final_samples).plot(kind='density',ax=ax[1,idx],lw=5, color=colors[cid], label=lbl, legend=False)#,label='Kurtosis: {:.3f} '.format(kurtosis(final_samples)))
        kurts_conv1 += [kurtosis(final_samples)]
        
        
        #ax.set_xlim([-20,20])
        ax[0,0].set_ylim([0.007, 0.15])
        ax[0,1].set_ylim([0.007, 0.15])
        ax[1,0].set_ylim([-1e-4,0.007])
        ax[1,1].set_ylim([-1e-4,0.007])
        
        
        ax[0,0].set_xlim([-100,100])
        ax[0,1].set_xlim([-100,100])
        ax[1,0].set_xlim([-100,100])
        ax[1,1].set_xlim([-100,100])
        
            
        #ax[1].set_xlim([-8,8])
        ax[0,1].set_ylabel('')
        ax[1,1].set_ylabel('')

        ax[0,0].set_xticks([])
        ax[0,1].set_xticks([])
        ax[0,1].set_yticks([])
        ax[1,1].set_yticks([])
        
        #ax[0].set_title(title[0])
            
        #if cid == 0:
        #ax[0].set_title(title[0])
        #ax[0].set_title(title[0])
        #ax[1].set_title(title[1])
        #ax[0, cid].set_title(m)

        kc1 = ['Kurtosis: {:.3f}'.format(k) for k,m in zip(kurts_conv1,lbl)]
        kc2 = ['Method: {}'.format(m) for k,m in zip(kurts_conv1,lbl)]
        #kc = ['Kurtosis: {:.3f}'.format(k) for k,m in zip(kurts_conv1, methods_)]
        #kf = ['Kurtosis: {:.3f} | Method: {}'.format(k,m) for k,m in zip(kurts_fc1, methods_)]

        ax[0,0].legend(labels,loc=3)
        ax[0,1].legend(labels,loc=1)
        #ax[1,0].legend(False)
        #ax[1,1].legend(loc=1)

        #ax[1].legend(kf)
        #ax[1].legend(kc1, loc=1)
        #
        #ax[0].legend()
        #ax[1].legend()
        
        plt.savefig(output_path+'/'+output_file, bbox_inches='tight')
        #'''
