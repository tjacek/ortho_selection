import numpy as np
from inliners.knn import get_inliners
import ens,tools,learn,reduction,files

def show_inliners(paths,out_path):
    data=tools.combined_dataset(paths[0],paths[1],True) 
    full_data,deep_data=data[0],data[2]
    helpers=get_inliners(full_data)
    show_template(full_data,out_path,helpers)

def show_template(datasets,out_path,helpers):
    files.make_dir(out_path)
    for i,date_i in enumerate(datasets):
        type_i= helpers[i]#lambda j,y_j: inliners[j][i]
        out_i="%s/nn%d" % (out_path,i)
        plot_i=reduction.tsne_plot(date_i,show=False,color_helper=type_i)  
        plot_i.savefig(out_i,dpi=1000)
        plot_i.close()

def inliner_ens(paths):
    data=tools.combined_dataset(paths[0],paths[1],True) 
    full_data,deep_data=data[0],data[2]
    inliners=get_inliners(full_data)
    result=inliner_voting(full_data,inliners)
#    result=base_voting(full_data)
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
    for i,vote_i in enumerate(votes):
        in_i=np.array([ inliners[j](i,cat_j) 
                    for j,cat_j in enumerate(vote_i)])
        y_pred.append(get_cat(vote_i,in_i))
    name=results[2]
    y_true=results[0][0]
    return y_true,y_pred,name

def get_cat(vote_i,in_i):
    if(np.sum(in_i)!=0):
        s_vote_i=[vote_ij 
            for vote_ij,in_ij in zip(vote_i,in_i)
                if(in_ij==1)]
    else:
        s_vote_i=vote_i
    s_vote_i=np.sum(learn.to_one_hot(s_vote_i),axis=0)
    return np.argmax(s_vote_i)
