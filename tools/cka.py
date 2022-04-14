# adapted from https://github.com/GitBl/analysisTB/blob/master/utils/CKA.py
import torch.nn as nn
import math
from torch import cuda
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import os

# According to paper - Include parts from https://github.com/google-research/google-research/tree/master/representation_similarity

def plot_CKA(matrix,xticks=None,yticks=None,figsize=(12,12),title="CKA matrix",output_path="./cka_analysis",output_file="cka.pdf",xlabel=None, ylabel=None):
    if (type(matrix) == torch.Tensor):
        matrix = matrix.detach().numpy().transpose()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.xaxis.set_ticks_position('bottom')
    if(xticks is not None):
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks,rotation=90)
        ax.set_yticks(range(len(yticks)) if yticks is not None else range(len(xticks)))
        ax.set_yticklabels(yticks if yticks is not None else xticks)
        if(xlabel is not None):
            ax.set_xlabel(xlabel)
        if(ylabel is not None):
            ax.set_ylabel(ylabel)
        
    else:
        ax.set_xlabel("Layer n°" if xlabel is None else xlabel)
        ax.set_ylabel("Layer n°" if ylabel is None else ylabel)
    
    cbar = fig.colorbar(cax)
    cbar.set_label("CKA value between layer")
    if(output_file is not None):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(output_path+'/'+output_file, bbox_inches='tight')
    plt.show()

def kernelize(X, cbf=True, sigma=1):
    # if (type(X) == torch.Tensor):
    #        X = X.detach().numpy()

    if(cbf):

        proj_mat = X.dot(X.T)
        first_part = np.diag(proj_mat) - proj_mat
        final_mat = first_part + first_part.T
        final_mat *= -1/(2*(sigma ** 2))
        #print("Final mat : {}".format(final_mat))
        return np.exp(final_mat)
    else:
        #return X.dot(X.T)
        return X.mm(torch.transpose(X, 0, 1))


def gram_centering(gram):
    if(type(gram) == torch.Tensor):
        means = torch.mean(gram, 0, dtype=torch.float32)
        means -= torch.mean(means) / 2
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]
    return gram

def googlegram_rbf(x, threshold=1.0):
    if (type(x) == torch.Tensor):
        dot_products = x.mm(torch.transpose(x, 0, 1))
        sq_norms = torch.diag(dot_products)
        sq_distances = -2 * dot_products + \
            sq_norms[:, None] + sq_norms[None, :]
        sq_median_distance = torch.median(sq_distances)
        return torch.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))
    else:
        dot_products = x.dot(x.T)

def CKA(X, Y, cbf, sigma=0.4, verbose=False):
    if(cbf):#use google's cbf
        K = gram_centering(googlegram_rbf(X,threshold=sigma))
        L = gram_centering(googlegram_rbf(Y,threshold=sigma))
    else:
        K = gram_centering(kernelize(X,cbf,sigma))
        L = gram_centering(kernelize(Y,cbf,sigma))

    numerator = HSIC(K, L)

    first_h = HSIC(K, K)
    second_h = HSIC(L, L)

    if verbose:
        print("{}, {}, {}".format(numerator, first_h, second_h))
        if(first_h == 0):
            print("K : {}".format(K))
            print("pure K : {}".format(kernelize(X, cbf, sigma)))
            print(X)
        if(second_h == 0):
            print("L : {}".format(L))
            print(Y)
    if(first_h == 0 or second_h == 0 or numerator == 0):
        return 0
    if(type(first_h) == torch.Tensor):
        return numerator/(torch.sqrt(first_h * second_h))
        del X
        del Y
    else:
        return numerator/(np.sqrt(first_h * second_h))


def HSIC(K, L):
    n = K.shape[0]

    if(type(K) == torch.Tensor):
        H = torch.from_numpy(np.eye(n) - np.ones((n, n))/n).float().to(K.device)
        final_mat = (K.mm(H)).mm(L.mm(H))

        return (torch.trace(final_mat))  # /(n-1)**2)
    else:
        H = np.eye(n) - np.ones((n, n))/n
        final_mat = (K.dot(H)).dot(L.dot(H))

        return (np.trace(final_mat))  # /(n-1)**2)


def CKA_1net(network, dataset, cbf=True, sigma=1, verbose=False, fast_computation = False, iteration_limit = 10, eligible_layers = [nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear]):
    """
    Returns the CKA matrix for the input networks, thanks to the Google algorithms.
    Excpect a matrix of size (nn.Conv layers size*nn.Conv layers size)
    cbf: Whether or not to use RBF kernel
    sigma: which sigma to use for the RBF kernel
    fast_computation: take only "iteration_limit" batchs for early results
    """
    print("INFO: Usage of CKA_1net is discouraged use eval_CKA instead")
    if (next(network.parameters()).is_cuda): #CUDA Trick
        network = network.cpu()
    network.eval()
    
    layer_names = []
    linking_list = []

    for name, module in network.named_modules():
        if type(module) in eligible_layers:
            linking_list.append(module)
            layer_names.append(name)

    hook_value = [-1]*len(linking_list)

    n = len(linking_list)

    def registering_hook(self, in_val, out_val):
        to_store = in_val[0]
        to_store = to_store.view(
            to_store.shape[0], np.product(to_store.shape[1:]))
        hook_value[linking_list.index(self)] = to_store

    hooks = []
    for module in network.modules():
        if type(module) in eligible_layers:
            hooks.append(module.register_forward_hook(registering_hook))

    return_matrix = torch.zeros((n, n))

    # Dataset pass
    if(fast_computation):
        iteration = 0
        for batch, _ in dataset:
            if(iteration<iteration_limit):
                iteration += 1
                with torch.no_grad():
                    network(batch)
                for i in range(n):
                    for j in range(i+1):
                        # print(hook_value[i])
                        temp = CKA(hook_value[i], hook_value[j],
                                   cbf, sigma, verbose)/iteration_limit
                        return_matrix[i][j] += temp
                        del temp
                print("Done: {:.2f}".format(100*(iteration/(iteration_limit))), end='\r')
        for i in range(n):
            for j in range(i, n):
                return_matrix[i, j] = return_matrix[j, i]
    else:
        iteration = 0
        for batch, _ in dataset:
            iteration += 1
            with torch.no_grad():
                network(batch)
            for i in range(n):
                for j in range(i+1):
                    # print(hook_value[i])
                    temp = CKA(hook_value[i], hook_value[j],
                               cbf, sigma, verbose)/len(dataset)
                    return_matrix[i][j] += temp
                    del temp
            print("Done: {:.2f}".format(100*(iteration/(len(dataset)+1))), end='\r')
        for i in range(n):
            for j in range(i, n):
                return_matrix[i, j] = return_matrix[j, i]
    for h in hooks:
        h.remove()

    return return_matrix, layer_names

def CKA_2net(network1, network2, dataset1, dataset2=None, cbf=True, sigma=1, verbose=False, fast_computation = False, iteration_limit = 10, eligible_layers = [nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear]):
    """
    dataset2 allows to pass a second dataloader with potentially a different Normalization etc- random shuffling and random transformations should be disabled in this case
    Returns the CKA matrix for the input networks, thanks to the Google algorithms.
    Excpect a matrix of size (nn.Conv layers size*nn.Conv layers size)
    cbf: Whether or not to use RBF kernel
    sigma: which sigma to use for the RBF kernel
    fast_computation: take only "iteration_limit" batchs for early results
    """
    print("INFO: Usage of CKA_2net is discouraged use eval_CKA instead")
    if (next(network1.parameters()).is_cuda):
        network1 = network1.cpu()
    if (next(network2.parameters()).is_cuda):
        network2 = network2.cpu()
    assert(dataset2 is None or len(dataset2)==len(dataset1))
    
    network1.eval()
    network2.eval()
    
    layer_names1 = []
    layer_names2 = []
    linking_list1 = []
    linking_list2 = []

    
    for name, module in network1.named_modules():
        if type(module) in eligible_layers:
            linking_list1.append(module)
            layer_names1.append(name)
    
    for name,module in network2.named_modules():
        if type(module) in eligible_layers:
            linking_list2.append(module)
            layer_names2.append(name)

    hook_value1 = [-1]*len(linking_list1)
    hook_value2 = [-1]*len(linking_list2)

    n1 = len(linking_list1)
    n2 = len(linking_list2)

    def registering_hook1(self, in_val, out_val):
        to_store = in_val[0]
        to_store = to_store.view(
            to_store.shape[0], np.product(to_store.shape[1:]))
        hook_value1[linking_list1.index(self)] = to_store
    
    def registering_hook2(self, in_val, out_val):
        to_store = in_val[0]
        to_store = to_store.view(
            to_store.shape[0], np.product(to_store.shape[1:]))
        hook_value2[linking_list2.index(self)] = to_store
    
    hooks = []

    for module in network1.modules():
        if type(module) in eligible_layers:
            hooks.append(module.register_forward_hook(registering_hook1))

    for module in network2.modules():
        if type(module) in eligible_layers:
            hooks.append(module.register_forward_hook(registering_hook2))

    return_matrix = torch.zeros((n1, n2))

    # Dataset pass
    if(fast_computation):
        iteration = 0
        for x in dataset1 if dataset2 is None else zip(dataset1,dataset2):
            if(dataset2 is None):
                batch1, _ = x
                batch2 = batch1
            else:
                (batch1, _), (batch2, _) = x
                
            if(iteration<iteration_limit):
                iteration += 1
                with torch.no_grad():
                    network1(batch1)
                    network2(batch2)
                for i in range(n1):
                    for j in range(n2):
                        # print(hook_value[i])
                        temp = CKA(hook_value1[i], hook_value2[j],
                                   cbf, sigma, verbose)/iteration_limit
                        return_matrix[i][j] += temp
                        del temp
                print("Done: {:.2f}".format(100*(iteration/(iteration_limit))), end='\r')
    else:
        iteration = 0
        for x in dataset1 if dataset2 is None else zip(dataset1,dataset2):
            if(dataset2 is None):
                batch1, _ = x
                batch2 = batch1
            else:
                (batch1, _), (batch2, _) = x
                
            iteration += 1
            with torch.no_grad():
                network1(batch1)
                network2(batch2)
            for i in range(n1):
                for j in range(n2):
                    # print(hook_value[i])
                    temp = CKA(hook_value1[i], hook_value2[j],
                               cbf, sigma, verbose)/len(dataset1)
                    return_matrix[i][j] += temp
                    del temp
            print("Done: {:.2f}".format(100*(iteration/(len(dataset1)+1))), end='\r')
    
    for h in hooks:
        h.remove()
        
    return return_matrix, layer_names1, layer_names2

def eval_CKA(network1, dataloader1, network2=None, dataloader2=None, cbf=False, sigma=0.5, verbose=False, num_batches = 0, eligible_layers = [nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear], device="cpu"):
    """
    dataloader2 allows to pass a second dataloader with potentially a different Normalization etc
    Returns the CKA matrix for the input networks, thanks to the Google algorithms.
    Excpect a matrix of size (nn.Conv layers size*nn.Conv layers size)
    cbf: Whether or not to use RBF kernel
    sigma: which sigma to use for the RBF kernel
    """
    device = torch.device(device)
    two_networks = network2 is not None
    two_dataloaders = dataloader2 is not None
    if(two_dataloaders):
        print("WARNING: You intend to use two separate dataloaders, make sure to turn off shuffling and disable random transformations in this case")

    network1 = network1.to(device)
    network1.eval()
    layer_names1 = []
    linking_list1 = []
  
    for name, module in network1.named_modules():
        if type(module) in eligible_layers:
            linking_list1.append(module)
            layer_names1.append(name)
            
    hook_value1 = [-1]*len(linking_list1)
    n1 = len(linking_list1)
    
    def registering_hook1(self, in_val, out_val):
        to_store = in_val[0]
        to_store = to_store.view(
            to_store.shape[0], np.product(to_store.shape[1:]))
        hook_value1[linking_list1.index(self)] = to_store
    
    hooks = []

    for module in network1.modules():
        if type(module) in eligible_layers:
            hooks.append(module.register_forward_hook(registering_hook1))

    if(two_networks):
        network2 = network2.to(device)
        network2.eval()
        layer_names2 = []
        linking_list2 = []
    
        for name,module in network2.named_modules():
            if type(module) in eligible_layers:
                linking_list2.append(module)
                layer_names2.append(name)
    
        hook_value2 = [-1]*len(linking_list2)
        n2 = len(linking_list2)
        
        def registering_hook2(self, in_val, out_val):
            to_store = in_val[0]
            to_store = to_store.view(
                to_store.shape[0], np.product(to_store.shape[1:]))
            hook_value2[linking_list2.index(self)] = to_store   
   
        for module in network2.modules():
            if type(module) in eligible_layers:
                hooks.append(module.register_forward_hook(registering_hook2))

    return_matrix = torch.zeros((n1, n2 if two_networks else n1)).to(device)

    dataloader = zip(dataloader1,dataloader2) if two_dataloaders else dataloader1
    dataloader = iter(dataloader)
    num_iterations = len(dataloader1) if num_batches==0 else min(len(dataloader1),num_batches)
    for batch_id in tqdm(list(range(num_iterations))):
        x = next(dataloader)
        if(two_dataloaders):
            (batch1, _), (batch2, _) = x
        else:
            batch1, _ = x
            batch2 = batch1
        
        with torch.no_grad():
            network1(batch1.to(device))
            if(two_networks):
                network2(batch2.to(device))

            for i in range(n1):
                for j in range(n2 if two_networks else i+1):
                    # print(hook_value[i])
                    temp = CKA(hook_value1[i], hook_value2[j] if two_networks else hook_value1[j],
                                   cbf, sigma, verbose)/num_iterations
                    return_matrix[i][j] += temp
                    del temp
            if(two_networks is False):
                for i in range(n1):
                    for j in range(i, n1):
                        return_matrix[i, j] = return_matrix[j, i]
    
    for h in hooks:
        h.remove()
    
    if(two_networks):
        return return_matrix.cpu(), layer_names1, layer_names2
    else:
        return return_matrix.cpu(), layer_names1