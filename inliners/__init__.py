import numpy as np
from inliners.knn import get_inliners
import ens,tools,learn,reduction,files
import clf

class InlinerEnsemble(object):
    def __init__(self,k=5):
        self.k=5

    def __call__(self,in_path,binary=True,clf="LR",acc_only=False,out_path=None):
        data=tools.combined_dataset(in_path[0],in_path[1],binary) 
        inliners=get_inliners(data[0],self.k)
        result=inliner_voting(data[0],inliners,binary,clf)
        ens.show_report(result,acc_only)

def show_inliners(paths,out_path):
    data=tools.combined_dataset(paths[0],paths[1],True) 
    full_data,deep_data=data[0],data[2]
    helpers=get_inliners(full_data,k=5)
    show_template(full_data,out_path,helpers)

def show_template(datasets,out_path,helpers):
    files.make_dir(out_path)
    for i,date_i in enumerate(datasets):
        type_i= helpers[i]#lambda j,y_j: inliners[j][i]
        out_i="%s/nn%d" % (out_path,i)
        plot_i=reduction.tsne_plot(date_i,show=False,color_helper=type_i)  
        plot_i.savefig(out_i,dpi=1000)
        plot_i.close()

#def inliner_ens(paths):
#    data=tools.combined_dataset(paths[0],paths[1],True) 
#    full_data,deep_data=data[0],data[2]
#    inliners=get_inliners(full_data)
#    result=inliner_voting(full_data,inliners)
#    ens.show_report(result)

def base_voting(datasets):
    votes=learn.make_votes(datasets,True,"LR")
    y_true=votes[0][0]
    y_pred=learn.voting(votes,True)
    return [y_true,y_pred,votes[0][2]]  

def inliner_voting(full_data,inliners,binary=True,clf="LR"):
    results=learn.make_votes(full_data,binary,clf)
    votes=learn.get_prob(results).T
    if(not binary):
        votes=np.swapaxes(votes,0,1)
    y_pred=[]
    for i,vote_i in enumerate(votes):
        if(binary):
            cat_i=binary_voting(i,vote_i, inliners)
        else:
            cat_i=prob_voting(i,vote_i, inliners)
        y_pred.append(cat_i)
    name=results[2]
    y_true=results[0][0]
    return y_true,y_pred,name

def prob_voting(i,vote_i, inliners):
    cats=np.argmax(vote_i,axis=0)
    in_i=np.array([ inliners[j](i,cat_j) 
                    for j,cat_j in enumerate(cats)])
    s_vote_i=[ vote_i[:,j] for j,in_j in enumerate(in_i)
                    if(in_j==1)]
    if(len(s_vote_i)<3):
        s_vote_i=vote_i
    s_vote_i=np.sum(s_vote_i,axis=0)
    print(s_vote_i)
    return np.argmax(s_vote_i)

def binary_voting(i,vote_i, inliners):
    in_i=np.array([ inliners[j](i,cat_j) 
                    for j,cat_j in enumerate(vote_i)])
    s_vote_i=select_votes(vote_i,in_i)
#    print(s_vote_i)    
    s_vote_i=np.sum(learn.to_one_hot(s_vote_i),axis=0)
    return np.argmax(s_vote_i)

def select_votes(vote_i,in_i):
#    print(vote_i)
    if(np.sum(in_i)>1):
        s_vote_i=[vote_ij 
            for vote_ij,in_ij in zip(vote_i,in_i)
                if(in_ij==1)]
        if(each_unique(s_vote_i)):
            return vote_i
        return s_vote_i
    return vote_i

def each_unique(s_vote_i):
    s=list(set(s_vote_i))
    return len(s)==len(s_vote_i)