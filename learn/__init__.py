import numpy as np
import feats
from sklearn.metrics import accuracy_score
import learn.clf 

def simple_exp(data):
    y_true,y_pred,names=train_model(data)
    return accuracy_score(y_true,y_pred)

def train_model(data,binary=True,clf_type="LR"):
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
    return y_true,y_pred,data.info

def ensemble_exp(datasets,binary=True):
    if(type(datasets)==str):
        datasets=feats.read(datasets)
    results=[train_model(data_i,binary) for data_i in datasets]
    y_true=results[0][0]
    if(binary):
        votes=np.array([to_one_hot(result_i[1]) for result_i in results])
    else:
        votes=np.array([result_i[1] for result_i in results])
    votes=np.sum(votes,axis=0)
    y_pred=[np.argmax(vote_i) for vote_i in votes]
    return accuracy_score(y_true,y_pred)


def combined_dataset(common_path,deep_path):
    common_data=feats.read(common_path)[0]
    deep_data=feats.read(deep_path)
    datasets=[common_data+data_i 
                for data_i in deep_data]
    return datasets

def to_one_hot(y):
    n_cats=max(y)+1
    one_hot=[]    
    for y_i in y:
        vec_i=np.zeros(n_cats)
        vec_i[y_i]=1
        one_hot.append(vec_i)
    return np.array(one_hot)

if __name__=="__main__":
    printf(ensemble_exp("../ens5/sim/feats",False))