import numpy as np
import ens,learn.report,tools,files
from learn import get_acc

def show_acc(in_path):
    paths=files.bottom_dict(in_path)
    result=[]
    all_acc=[]
    for path_i in paths:
        for binary_i in [True,False]:
            result_i=ens.read_result(path_i,binary_i,"raw")
            y_true,y_pred=result_i[0],result_i[1]
            n_cat= max(result_i[0])+1
            acc_i=[learn.report.binary_acc(y_true,y_pred,cat_i)
                    for cat_i in range(n_cat)]
            all_acc.append(acc_i)
    return all_acc

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

print(show_acc("votes/MSR"))
#ens_stats(paths)
#show_votes(paths)