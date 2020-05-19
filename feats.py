import os.path
import numpy as np
from sklearn import preprocessing
import files,tools

class FeatureSet(object):
    def __init__(self,X,info):
        self.X=X
        self.info=info

    def __len__(self):
        return len(self.info)

    def dim(self):
        return self.X.shape[1]

    def n_cats(self):
        return len(set(self.get_labels()))

    def get_labels(self):
        return [ int(info_i.split('_')[0])-1 for info_i in self.info]

    def to_dict(self):
        return { self.info[i]:x_i 
                    for i,x_i in enumerate(self.X)}

    def split(self,selector=None):
        feat_dict=self.to_dict()
        train,test=tools.split(feat_dict,selector)
        return from_dict(train),from_dict(test)

    def norm(self):
        self.X=preprocessing.scale(self.X)
        return self
        
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