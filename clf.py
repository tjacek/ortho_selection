import numpy as np
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
import feats,learn

def person_selection(common_path,deep_path):
    datasets=learn.combined_dataset(common_path,deep_path)
    acc=[pred_person(data_i) for data_i in datasets]
    acc=np.array(acc)
    print(acc)
    acc=(acc-np.mean(acc))/np.std(acc)
    print(acc)
    datasets=[data_i 
             for i,data_i in enumerate(datasets)
                if(acc[i]> -1)]
    acc=learn.ensemble_exp(datasets,binary=False,clf="LR",acc_only=True)
    print(acc)

def pred_person(data_i):
    train=data_i#,test=data_i.split()
    y=[ "%s_%d" % (name_i.split('_')[1],i) 
            for i,name_i in enumerate(train.info)]
    person_i=feats.FeatureSet(train.X,y)   
    print(person_i.n_cats())

    return learn.train_model(person_i,
                binary=True,clf_type="LR",acc_only=True)

#    data_i.norm()
#    train=data_i.split()[0]	
#    train.info=person_cats(train.info)
#    model=LogisticRegression(solver='liblinear')
#    model.fit(train.X,train.get_labels())
#    y_true=train.get_labels()
#    y_pred=model.predict(train.X)
#    return accuracy_score(y_true,y_pred)

person_selection("../proj2/stats/feats","../ens5/basic/feats")