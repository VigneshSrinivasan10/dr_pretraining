import numpy as np
import pandas as pd
from pathlib import Path
import os

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

import random

from .stratify import split_stratified

from PIL.ImageStat import Stat

imagenet_stats = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]
cifar_stats = [[0.491, 0.482, 0.447],[0.247, 0.243, 0.262]]
chexpert_stats = [[0.50283729, 0.50283729, 0.50283729], [0.29132762, 0.29132762, 0.29132762]]
dr_stats = [[0.3203, 0.2244, 0.1609], [0.2622, 0.1833, 0.1318]]


##############################################################################
#UTILS
##############################################################################

def split_stratified_wrapper(df_train,fractions,dataset,save_path,col_subset,filename_postfix="subset",disregard_patients=False,disregard_labels=False):
    filename_split = os.path.join(save_path,dataset+"_"+filename_postfix+".txt")
    if(disregard_labels):
        df_train["dummy"]=0

    if dataset.startswith("chexpert") or dataset=="mimic_cxr" or dataset=="cxr14" or dataset=="cxr_combined": #cxr
        return split_stratified(df_train,fractions,filename_split,col_subset,col_index="image_id",col_label="dummy" if disregard_labels else "label",col_group=None if disregard_patients else "patient_id",label_multi_hot=True)
    elif dataset == "diabetic_retinopathy":
        return split_stratified(df_train,fractions,filename_split,col_subset,col_label="dummy" if disregard_labels else "label",col_group=None if disregard_patients else "patient")
    else:#default for image datasets
        return split_stratified(df_train,fractions,filename_split,col_subset,col_label="dummy" if disregard_labels else "label")


#from fastaiv1
def _get_files(parent, p, f, extensions):
    p = Path(p)
    if isinstance(extensions,str): extensions = [extensions]
    low_extensions = [e.lower() for e in extensions] if extensions is not None else None
    res = [p/o for o in f if not o.startswith('.')
           and (extensions is None or o.split(".")[-1].lower() in low_extensions)]
    return res

def get_files(path, extensions=None, recurse=False, exclude=None,
              include=None, followlinks=False):
    "Return list of files in `path` that have a suffix in `extensions`; optionally `recurse`."
    if recurse:
        res = []
        for i,(p,d,f) in enumerate(os.walk(path, followlinks=followlinks)):
            # skip hidden dirs
            if include is not None and i==0:   d[:] = [o for o in d if o in include]
            elif exclude is not None and i==0: d[:] = [o for o in d if o not in exclude]
            else:                              d[:] = [o for o in d if not o.startswith('.')]
            res += _get_files(path, p, f, extensions)
        return res
    else:
        f = [o.name for o in os.scandir(path) if o.is_file()]
        res = _get_files(path, path, f, extensions)
        return res

##############################################################################
#GENERIC DATASET
##############################################################################        
class ImageDataframeDataset(Dataset):
    '''creates a dataset based on a given dataframe'''
    def __init__(self, df, transform=None, target_transform=None,
                     loader=default_loader,col_filename="path", col_target="label",
                     col_target_set=None):
        super(ImageDataframeDataset).__init__()
        self.col_target_set = col_target_set
        if(col_target_set is not None):#predefined set for semi-supervised
            self.samples = list(zip(np.array(df[col_filename]), np.array(df[col_target]), np.array(df[col_target_set],dtype=np.int8)))
        else:
            self.samples = list(zip(np.array(df[col_filename]), np.array(df[col_target])))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if(self.col_target_set is not None):
            path, target, subset = self.samples[index]
        else:
            path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if(self.col_target_set is not None):
            return sample, [target, subset]
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)



##############################################################################
#PREPARING DATAFRAMES
##############################################################################
def prepare_imagefolder_df(root,label_itos=None,extensions=["jpg","jpeg","png"]):
    '''prepares dataframe for imagefolder dataset'''
    files = get_files(root,extensions=extensions,recurse=True)
    df=pd.DataFrame(files,columns=["path"])
    df["label_raw"]=df.path.apply(lambda x: x.parts[-2])
    if(label_itos is None):
        label_itos = np.unique(df["label_raw"])
    label_stoi = {s:i for i,s in enumerate(label_itos)}
    df["label"] = df.label_raw.apply(lambda x:label_stoi[x])
    print(root,":",len(df),"samples.")
    return df,label_itos

##############################################################################
#DIABETIC RETINOPATHY
##############################################################################

def prepare_diabetic_retinopathy_df(root, args, label_itos = [*map(str,range(5))]):
    root = Path(root)

    df_train = pd.read_csv(root/"trainLabels.csv")
    df_train["patient"]=df_train["image"].apply(lambda x: x.split("_")[0])
    df_train["left_eye"]=df_train["image"].apply(lambda x: x.split("_")[1]=="left")
    df_train["label"]=df_train["level"]
    df_train["path"]=df_train["image"].apply(lambda x:root/"train"/(x+".jpeg"))

    df_valid = pd.read_csv(root/"retinopathy_solution.csv")
    df_valid["patient"]=df_valid["image"].apply(lambda x: x.split("_")[0])
    df_valid["left_eye"]=df_valid["image"].apply(lambda x: x.split("_")[1]=="left")
    df_valid["label"]=df_valid["level"]
    df_valid["path"]=df_valid["image"].apply(lambda x:root/"test"/(x+".jpeg"))

    ### Randomly select 33423 samples from test set
    import random
    random.seed(1)
    valid_idxs = random.sample(range(len(df_train)), 5126)
    train_idxs = list(set(range(len(df_train))) - set(valid_idxs))
    
    df_test = df_valid
    df_valid = df_train.loc[valid_idxs]
    df_train = df_train.loc[train_idxs]
    
    print(root,"train:",len(df_train),"samples.")    
    print(root,"valid:",len(df_valid),"samples.")
    print(root,"test:",len(df_test),"samples.")
    
    return df_train, df_valid, df_test, label_itos

def prepare_binary_classification(df, eval_dr='binary_rdr'):
    if eval_dr == 'binary_rdr':
        df["label"].replace(
            to_replace=[0, 1, 2],
            value=0,
            inplace=True
        )   
        df["label"].replace(
            to_replace=[3, 4],
            value=1,
            inplace=True
        )
        
    elif eval_dr == 'binary_norm':
        df["label"].replace(
            to_replace=[2, 3, 4],
            value=1,
            inplace=True
        )   
    elif eval_dr == 'binary_dme':
        df["label"].replace(
            to_replace=[0, 1],
            value=0,
            inplace=True
        )   

        df["label"].replace(
            to_replace=[2, 3, 4],
            value=1,
            inplace=True
        )

    return df

def prepare_messidor_df(root, label_itos = [*map(str,range(5))]):
    root = Path(root)

    df_test = pd.read_csv(root/"messidor_data.csv")
    # import pdb;
    # pdb.set_trace()
    df_test["patient"]="IMAGES/"+df_test["image_id"]
    df_test["label"]=df_test["adjudicated_dr_grade"]
    #df_test["binary_label"]=df_test["adjudicated_dme"] # Referrable vs Non-Referrable
    df_test["path"]=df_test["image_id"].apply(lambda x:root/"IMAGES"/x)

    shape = df_test.shape[0]
    df_test = df_test[df_test['adjudicated_dr_grade'].notna()]
    print("Number of test samples removed: {}".format(df_test.shape[0] - shape))

    df_train = df_test.copy() #dummy
    
    print(root,"test:",len(df_test),"samples.")

    return df_train, df_test, label_itos

def prepare_messidor_1_df(root, label_itos = [*map(str,range(5))]):
    root = Path(root)

    df_test = pd.read_excel(root/"test/test.xls")
    # import pdb;
    # pdb.set_trace()
    df_test["patient"]="test/"+df_test["Image name"]
    df_test["label"]=df_test["Retinopathy grade"]
    #df_test["binary_label"]=df_test["adjudicated_dme"] # Referrable vs Non-Referrable
    df_test["path"]=df_test["Image name"].apply(lambda x:root/"test"/x)

    df_train = pd.read_excel(root/"train/train.xls")
    # import pdb;
    # pdb.set_trace()
    df_train["patient"]="train/"+df_train["Image name"]
    df_train["label"]=df_train["Retinopathy grade"]
    #df_test["binary_label"]=df_test["adjudicated_dme"] # Referrable vs Non-Referrable
    df_train["path"]=df_train["Image name"].apply(lambda x:root/"train"/x)

    df_test = pd.concat([df_train,df_test],ignore_index=True)
    
    print(root,"train:",len(df_train),"samples.")
    print(root,"test:",len(df_test),"samples.")
    
    return df_train, df_test, label_itos

##############################################################################
#GETTING DATATSET STATS
##############################################################################
def stats_from_ds(ds,div=255):
    means=[]
    stds=[]
    for d in tqdm(ds_train):
        stat = Stat(d[0])
        means.append(stat.mean)
        stds.append(stat.stddev)
    means = np.mean(means,axis=0)
    stds = np.mean(stds,axis=0)
    return [means/div,stds/div]

#chexpert_stats = stats_from_ds(ds_train)
