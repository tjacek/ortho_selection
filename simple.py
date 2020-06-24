import numpy as np
import feats,learn
import ens,clf

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
    ensemble=ens.Ensemble(clf.person_selection)
    ensemble(in_path,clf="LR")

def full_ensemble(common_path,deep_path):
    datasets=learn.combined_dataset(common_path,deep_path)
#    datasets=[basic_select(data_i) for data_i in datasets]
    acc=learn.ensemble_exp(datasets)
    print(acc)

paths=("../outliners/common/stats/feats","../outliners/ens/sim/feats")
ensemble_exp(paths)