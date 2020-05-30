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
    datasets=feats.read(in_path)
#    datasets=[basic_select(data_i) for data_i in datasets]
    acc=learn.ensemble_exp(datasets)
    print(acc)
    return acc

def full_ensemble(common_path,deep_path):
    datasets=learn.combined_dataset(common_path,deep_path)
#    datasets=[basic_select(data_i) for data_i in datasets]
    acc=learn.ensemble_exp(datasets)
    print(acc)


#ensemble_exp("../ens5/sim/feats")
#full_ensemble("../proj2/stats/feats","../ens5/basic/feats")
acc=learn.get_acc("../proj2/stats/feats","../ens5/sim/feats")
print(acc)