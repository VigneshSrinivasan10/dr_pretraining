import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def export_results(args_or_dict_lst,filename):
    '''exports list (or single instance) of dic or argparse.Namespace into json file'''
    if(not(isinstance(args_or_dict_lst,list))):
        args_or_dict_lst= [args_or_dict_lst]
    args_all = {}
    
    for a in args_or_dict_lst:
        if(isinstance(a,argparse.Namespace)):
            args_all.update(vars(a))
        elif(isinstance(a,dict)):
            args_all.update(a)
        else:
            assert(True)
    with open(filename, 'w') as fp:
        json.dump(args_all, fp,sort_keys=True, indent=4)


def collect_results(path=".",include_filename=True, include_modificationtime=True):
    '''reads all json files in path and combines them into pandas dataframe'''
    path = Path(path)
    results=[]
    for f in list(path.glob('**/*.json')):
        with open(f, 'r') as fp:
            d=json.load(fp)
            if(include_filename):
                d["json_filename"]= f
            if(include_modificationtime):
                d["json_modificationtime"]=datetime.fromtimestamp(f.stat().st_mtime)
            results.append(d)
        
    return pd.DataFrame(results)
