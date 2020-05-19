import os.path
import numpy as np
import files

class FeatureSet(object):
    def __init__(self,X,info):
        self.X=X
        self.info=info

    def __len__(self):
        return len(self.info)

def read(in_path):
    if(not os.path.isdir(in_path)):
        return [from_dict(read_single(in_path))]
    return [from_dict(read_single(path_i)) 
                for path_i in files.top_files(in_path)]

def from_dict(feat_dict):
    info=files.natural_sort(feat_dict.keys())
    X=np.array([feat_dict[info_i] 
                    for info_i in info])
    return FeatureSet(X,info)

def read_single(in_path):
    lines=open(in_path,'r').readlines()
    feat_dict={}
    for line_i in lines:
        raw=line_i.split('#')
        if(len(raw)>1):
            data_i,info_i=raw[0],raw[-1]
            info_i= files.clean_str(info_i)
            feat_dict[info_i]=np.fromstring(data_i,sep=',')
    return feat_dict