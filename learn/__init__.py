import numpy as np
import feats
from sklearn.metrics import accuracy_score
import learn.clf,tools

def train_model(data,binary=False,clf_type="LR",acc_only=False):
    if(type(data)==str):	
        data=feats.read(data)[0]
    data.norm()
    train,test=data.split()
    model= learn.clf.get_cls(clf_type)
    model.fit(train.X,train.get_labels())
    y_true=test.get_labels()
    if(binary):
        y_pred=model.predict(test.X)
    else:
        y_pred=model.predict_proba(test.X)
    if(acc_only):
        return accuracy_score(y_true,y_pred)
    else:
        return y_true,y_pred,test.info

def ensemble_exp(datasets,binary=False,clf="LR",acc_only=True):
    if(type(datasets)==str):
        datasets=feats.read(datasets)
    if(type(clf)==list):
        results=[]
        for clf_i in clf:
            results+=[train_model(data_i,binary,clf_i) for data_i in datasets]
    else:
        results=[train_model(data_i,binary,clf) for data_i in datasets]
    y_true=results[0][0]
    y_pred=voting(results,binary)
    if(acc_only):
        return accuracy_score(y_true,y_pred)
    return y_true,y_pred,results[0][2]

def voting(results,binary):
    if(binary):
        votes=np.array([to_one_hot(result_i[1]) for result_i in results])
    else:
        votes=np.array([result_i[1] for result_i in results])
    votes=np.sum(votes,axis=0)
    y_pred=[np.argmax(vote_i) for vote_i in votes]
    return y_pred

def to_one_hot(y):
    n_cats=max(y)+1
    one_hot=[]    
    for y_i in y:
        vec_i=np.zeros(n_cats)
        vec_i[y_i]=1
        one_hot.append(vec_i)
    return np.array(one_hot)

def get_acc(common_path,deep_path,clf="LR"):
    datasets=tools.combined_dataset(common_path,deep_path)
    return [train_model(data_i,True,clf,True) 
                for data_i in datasets]

if __name__=="__main__":
    printf(ensemble_exp("../ens5/sim/feats",False))