import numpy as np
import feats

def basic_select(data):
    if(type(data)==str):
        data=feats.read_single(data,as_dict=False)
    train,test=data.split()
    info=train.mutual()
    info=(info-np.mean(info))/np.std(info)
    new_X=[ x_i
                for i,x_i in enumerate(data.X.T)
                    if(info[i]>-1)]
    new_X=np.array(new_X).T
    new_feats=data.modify(new_X)
    print(new_feats.dim())
    return new_feats