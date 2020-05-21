import numpy as np
import feats,learn

def compare(in_path):
    old_acc=learn.simple_exp(in_path)
    new_data=select(in_path)
    new_acc=learn.simple_exp(new_data)
    print("%f,%f" % (old_acc,new_acc))

def select(in_path):
    data=feats.read_single(in_path,as_dict=False)
    train,test=data.split()
    info=train.mutual()
    info=(info-np.mean(info))/np.std(info)
    new_X=[ x_i
                for i,x_i in enumerate(data.X.T)
                    if(info[i]>-1)]
    new_X=np.array(new_X).T
    new_feats=data.modify(new_X)
    print(len(new_feats))
    return new_feats

compare("../proj2/stats/feats")