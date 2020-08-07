import numpy as np
from sklearn import svm
import ens,tools,learn

def inliner_ens(paths):
    common_path,deep_path=paths
    data=tools.combined_dataset(common_path,deep_path,sub_datasets=True) 
    full_data,deep_data=data[0],data[2]
    detectors=get_detector(deep_data)
    inliners=find_inliners(deep_data,detectors)
#    result=base_voting(full_data)
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

def find_inliners(deep_data,detectors):
    test_data=[ data_i.split()[1] for data_i in deep_data]
    inliners=[ detect_i.score_samples(test_i.X) 
                    for detect_i,test_i in zip(detectors,test_data)]
    inliners=np.array(inliners).T
    inliners[inliners>0]=1
    inliners=1-inliners
    return inliners

def get_detector(data_i):
    if(type(data_i)==list):
        return [get_detector(data) for data in data_i]
    train,test=data_i.split()	
    clf_i=svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf_i.fit_predict(train.X)
    return clf_i

paths=("../outliners/common/stats/feats","../outliners/ens/sim/feats")
inliner_ens(paths)