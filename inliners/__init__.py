import numpy as np
from inliners.knn import get_inliners
import ens,tools,learn,reduction,files
import clf

class InlinerEnsemble(object):
    def __init__(self,k=5):
        self.k=5

    def __call__(self,in_path,binary=False,clf="LR",acc_only=False,out_path=None):
        data=tools.read_datasets(in_path)
        inliners=get_inliners(data,self.k)
        result=inliner_voting(data,inliners,binary,clf,out_path)
        return ens.show_report(result,acc_only)

def inliner_voting(full_data,inliners,binary=True,clf="LR",out_path=None):
    results=ens.get_votes(full_data,binary,clf,out_path)
    votes=[result_i[1] for result_i in results]
    votes=np.swapaxes(votes,0,1)
    y_pred=[]
    for i,vote_i in enumerate(votes):
        cat_i=prob_voting(i,vote_i, inliners)
        y_pred.append(cat_i)
    name=results[0][2]
    y_true=results[0][0]
    return y_true,y_pred,name

def prob_voting(i,vote_i, inliners):
    cats=np.argmax(vote_i,axis=1)
    in_i=np.array([ inliners[j](i,cat_j) 
                    for j,cat_j in enumerate(cats)])
    s_vote_i=[ vote_j for j,vote_j in enumerate(vote_i)
                    if(in_i[j]==1)]
    s_vote_i=np.array(s_vote_i)
    if(len(s_vote_i)<3):
        s_vote_i=vote_i
    s_vote_i=np.sum(s_vote_i,axis=0)
    return np.argmax(s_vote_i)

def each_unique(s_vote_i):
    s=list(set(s_vote_i))
    return len(s)==len(s_vote_i)