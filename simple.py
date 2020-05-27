import numpy as np
import feats,learn

def compare(in_path):
    old_acc=learn.simple_exp(in_path)
    new_data=basic_select(in_path)
    new_acc=learn.simple_exp(new_data)
    print("%f,%f" % (old_acc,new_acc))

def ens_compare(in_path):
    old_acc=learn.ensemble_exp(in_path)
    new_acc=ensemble_exp(in_path)
    print("%f,%f" % (old_acc,new_acc))

def ensemble_exp(in_path):
#    datasets=feats.read(in_path)
    common_path='../proj2/stats/feats'
    datasets=learn.combined_dataset(common_path,in_path)
    datasets=[basic_select(data_i) for data_i in datasets]
    acc=learn.ensemble_exp(datasets)
    print(acc)
    return acc

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

ensemble_exp("../ens5/sim/feats")