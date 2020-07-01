import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import feats,learn,tools

def get_selection(selection_type):
    if(selection_type=="person"):
        return person_selection
    return simple_selection

def person_selection(in_path):
    deep_only=(None,in_path[1])
    datasets=tools.read_datasets(deep_only)
    acc=[pred_person(data_i) for data_i in datasets]
    acc=np.array(acc)
    print(acc)
    acc=(acc-np.mean(acc))/np.std(acc)
    print(acc)
    cond=lambda x:x>-1
    return selection_template(in_path,acc,cond)

def selection_template(in_path,acc,cond):
    datasets=tools.read_datasets(in_path)
    s_datasets=[data_i 
             for i,data_i in enumerate(datasets)
                if(cond(acc[i]))]
    if(len(s_datasets)!=0):
        datasets=s_datasets
    print("n_clf %d" % len(datasets))
    return datasets

def pred_person(data_i):
    train,test=data_i.split()
    y=[ "%s_%d" % (name_i.split('_')[1],i) 
            for i,name_i in enumerate(train.info)]
    person_i=feats.FeatureSet(train.X,y)   
    return person_train(data_i)
    
def person_train(data_i):
    data_i.norm()
    model=LogisticRegression(solver='liblinear')
    model.fit(data_i.X,data_i.get_labels())
    y_true=data_i.get_labels()
    y_pred=model.predict(data_i.X)
    return accuracy_score(y_true,y_pred)

def simple_selection(in_path):
    datasets=tools.read_datasets(in_path)
    acc=cross_acc(datasets)
    acc=np.array(acc)
    acc= (acc-np.mean(acc))/np.std(acc)
    print(acc)
    cond=lambda x:x>0
    return selection_template(in_path,acc,cond)

def cross_acc(datasets):
    datasets=[data_i.split()[0] for data_i in datasets]
    for data_i in datasets:
        data_i.info=["%s_%d" %(name_i.split('_')[0],i) 
            for i,name_i in enumerate(data_i.info)]
    acc=[ learn.train_model(data_i,
            binary=True,clf_type="LR",acc_only=True)
                for data_i in datasets]
    return acc

if __name__=="__main__":
    paths=("../outliners/common/stats/feats","../outliners/ens/sim/feats")