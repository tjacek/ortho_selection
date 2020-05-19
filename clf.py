import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import feats

def person_inv(in_path):
    datasets=feats.read(in_path)
    acc=[pred_person(data_i) for data_i in datasets]
    print(acc)
    return np.argsort(acc)

def pred_person(data_i):
    data_i.norm()
    train=data_i.split()[0]	
    train.info=person_cats(train.info)
    model=LogisticRegression(solver='liblinear')
    model.fit(train.X,train.get_labels())
    y_true=train.get_labels()
    y_pred=model.predict(train.X)
    return accuracy_score(y_true,y_pred)

def person_cats(names):
    return [ "%s_%d" %(name_i.split("_")[1],i) 
                for i,name_i in enumerate(names)]

ord=person_inv("../ens5/sim/feats")
print(ord)