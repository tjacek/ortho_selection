import numpy as np
import ens,learn.report,tools
from learn import get_acc

def show_acc(paths):
    ensemble=ens.get_ensemble()
    datasets=[ (paths['common'],paths['ens']),(None,paths['ens']),(paths['common'],None)]
    acc=[]
    for data_i in datasets:
        acc_i=ensemble(data_i,clf="LR",acc_only=True)
        acc.append(acc_i)
    print("***************")    
    for acc_i in acc:
        print(acc_i)

def show_votes(paths,binary=True):
    ensemble=ens.get_ensemble()
    paths=(paths['common'],paths['ens'])
    datasets=tools.read_datasets(paths)
    votes=ens.get_votes(datasets,binary,"LR",None)
    counter={name_j:np.zeros(20) for name_j in votes[0][2]}
    for y_true,y_pred,name_i in votes:
        y_pred=np.argmax(y_pred,axis=1)
        for y_j,name_j in zip(y_pred,name_i):
            counter[name_j][y_j]+=1
    print(counter)

def ens_stats(paths):
    result=learn.report.ens_acc((paths['common'],paths['ens']),clf="LR",acc_only=False)	
    result=[ [result_i[0], np.argmax(result_i[1],axis=1),result_i[2]] for result_i in result]
    full_result=simple_exp(paths)
    cat_dict=learn.report.cat_by_error(full_result,binary=True)
    print(learn.report.to_acc(result))
    print(cat_dict)
    for cat_i in cat_dict.keys():
        print(cat_i)
        acc_i=learn.report.cat_acc(result,cat_i)
        print(acc_i)

def get_paths(common_path,ens_path):
    return {'common':common_path,'ens':ens_path}

def simple_exp(paths):
    ensemble=ens.get_ensemble()
    path=(paths['common'],paths['ens'])
    return ensemble.get_result(path,clf="LR")

common_path="proj/MSR/scale/max_z/feats"
common_path1="proj/MSR/scale/stats/feats"

deep_path="proj/MSR/ens/basic/feats"
paths=get_paths([common_path,common_path1],deep_path)
#print(get_acc((paths['common'],paths['ens'])))
#show_acc(paths)
#ens_stats(paths)
show_votes(paths)
