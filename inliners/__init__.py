import numpy as np
from inliners.one_svm import get_inliners
import ens,tools,learn,reduction,files

def show_inliners(paths,out_path):
    data=tools.combined_dataset(paths[0],paths[1],True) 
    full_data,deep_data=data[0],data[2]
    inliners=get_inliners(deep_data)
    files.make_dir(out_path)
    for i,date_i in enumerate(full_data):
        type_i= lambda j,y_j: inliners[j][i]
        out_i="%s/nn%d" % (out_path,i)
        plot_i=reduction.tsne_plot(date_i,show=False,color_helper=type_i)  
        plot_i.savefig(out_i,dpi=1000)
        plot_i.close()

def inliner_ens(paths):
    data=tools.combined_dataset(paths[0],paths[1],True) 
    full_data,deep_data=data[0],data[2]
    inliners=get_inliners(deep_data)
    result=inliner_voting(full_data,inliners)
    ens.show_report(result)

def base_voting(datasets):
    votes=learn.make_votes(datasets,True,"LR")
    y_true=votes[0][0]
    y_pred=learn.voting(votes,True)
    return [y_true,y_pred,votes[0][2]]  

def inliner_voting(full_data,inliners):
    results=learn.make_votes(full_data,True,"LR")
    votes=learn.get_prob(results).T
    y_pred=[]
    for vote_i,in_i in zip(votes,inliners):
#        print(vote_i)
        s_vote_i=select_votes(vote_i, in_i)
        s_vote_i=np.sum(learn.to_one_hot(s_vote_i),axis=0)
        y_pred.append(np.argmax(s_vote_i)) 
    name=results[2]
    y_true=results[0][0]
    return y_true,y_pred,name

def select_votes(vote_i, in_i):
    return [vote_ij for vote_ij,in_ij in zip(vote_i,in_i)
	            if(in_ij==1)]
