import numpy as np
import ens,learn.report,tools,files
from learn import get_acc
from sklearn.metrics import accuracy_score

def show_acc(in_path,acc_type="cat"):
    paths=files.bottom_dict(in_path)
    result=[]
    all_acc=[]
    if(acc_type=="cat"):
        fun= cat_acc 
    elif(acc_type=="acc_indv"): 
        fun=indv_acc
    else: 
        fun=indv_cat
    for path_i in paths:
        for binary_i in [True,False]:
            acc_i=fun(path_i,binary_i)
            all_acc.append(acc_i)
    return all_acc

def cat_acc(path_i,binary_i):
    result_i=ens.read_result(path_i,binary_i,"raw")
    y_true,y_pred=result_i[0],result_i[1]
    n_cat= max(result_i[0])+1
    acc_i=[learn.report.binary_acc(y_true,y_pred,cat_i)
            for cat_i in range(n_cat)]
    return acc_i

def indv_acc(path_i,binary_i):
    votes=get_votes(path_i,binary_i)
    acc=[]
    for y_true,y_pred,names in votes:
        y_pred=np.argmax(y_pred,axis=1)
        acc_i= accuracy_score(y_true,y_pred)
        acc.append(acc_i)
    return acc

def indv_cat(path_i,binary_i):
    votes=get_votes(path_i,binary_i)
    n_cat=len(votes)
    n_cat=max(votes[0][0])+1
    acc=[]
    for y_true,y_pred,names in votes:
        y_pred=np.argmax(y_pred,axis=1)
        acc_i=[learn.report.binary_acc(y_true,y_pred,cat_i)
                for cat_i in range(n_cat)]
        acc.append(acc_i)
    return acc

def get_votes(path_i,binary_i):
    votes=learn.votes.read_votes(path_i)
    if(binary_i):
        votes=[learn.votes.as_binary(vote_i) 
                for vote_i in votes]
    return votes

#def show_votes(paths,binary=True):
#    ensemble=ens.get_ensemble()
#    paths=(paths['common'],paths['ens'])
#    datasets=tools.read_datasets(paths)
#    votes=ens.get_votes(datasets,binary,"LR",None)
#    counter={name_j:np.zeros(20) for name_j in votes[0][2]}
#    for y_true,y_pred,name_i in votes:
#        y_pred=np.argmax(y_pred,axis=1)
#        for y_j,name_j in zip(y_pred,name_i):
#            counter[name_j][y_j]+=1
#    print(counter)

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

if __name__ == '__main__':
    print(show_acc("votes/MSR",acc_type="indv_cat"))
#ens_stats(paths)
#show_votes(paths)