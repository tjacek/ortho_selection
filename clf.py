import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import feats,learn

def person_selection(common_path,deep_path):
    datasets=feats.read(deep_path)
    acc=[pred_person(data_i) for data_i in datasets]
    acc=np.array(acc)
    print(acc)
    acc=(acc-np.mean(acc))/np.std(acc)
    print(acc)
    s_ensemble(common_path,deep_path,acc)

def s_ensemble(common_path,deep_path,acc):
    datasets=learn.combined_dataset(common_path,deep_path)
    datasets=[data_i 
             for i,data_i in enumerate(datasets)
                if(acc[i]> -1.0)]
    print("n_clf %d" % len(datasets))    
    acc=learn.ensemble_exp(datasets,binary=False,clf="LR",acc_only=True)
    print(acc)

def pred_person(data_i):
    train,test=data_i.split()
    y=[ "%s_%d" % (name_i.split('_')[1],i) 
            for i,name_i in enumerate(train.info)]
    person_i=feats.FeatureSet(train.X,y)   
    return person_train(data_i)
#    return learn.train_model(person_i,
#                binary=True,clf_type="LR",acc_only=True)


def person_train(data_i):
    data_i.norm()
    model=LogisticRegression(solver='liblinear')
    model.fit(data_i.X,data_i.get_labels())
    y_true=data_i.get_labels()
    y_pred=model.predict(data_i.X)
    return accuracy_score(y_true,y_pred)

def simple_selection(common_path,deep_path):
#    datasets=feats.read(deep_path)
    datasets=learn.combined_dataset(common_path,deep_path)
    datasets=[data_i.split()[0] for data_i in datasets]
    for data_i in datasets:
        data_i.info=["%s_%d" %(name_i.split('_')[0],i) 
            for i,name_i in enumerate(data_i.info)]
    acc=[ learn.train_model(data_i,
            binary=True,clf_type="LR",acc_only=True)
                for data_i in datasets]
    acc=np.array(acc)
    acc= (acc-np.mean(acc))/np.std(acc)
    print(acc)
    s_ensemble(common_path,deep_path,acc)

simple_selection("../proj2/stats/feats","../ens5/basic/feats")