import numpy as np
import feats,tools

def basic_select(data):
    if(type(data)==str):
        data=feats.read_single(data,as_dict=False)
    train,test=data.split()
    info=train.mutual()
    info=(info-np.mean(info))/np.std(info)
    return select_feats(data,info,lambda f: f<1)

def person_select(data):
    train,test=data.split()
    train.info=tools.person_cats(train.info)
    print(train.n_cats())
    info=train.mutual()
    info=(info-np.mean(info))/np.std(info)
    return select_feats(data,info,lambda f: f<1)

def complex_select(data):
    train,test=data.split()
    info_cat=train.mutual()
    train.info=tools.person_cats(train.info)
    info_person=train.mutual()
    info= info_cat-info_person 
    info=(info-np.mean(info))/np.std(info)
    return select_feats(data,info,lambda f: f>-1)

def select_feats(data,info,cond):
    new_X=[ x_i
            for i,x_i in enumerate(data.X.T)
                if(info[i]>-1)]
    new_X=np.array(new_X).T
    new_feats=data.modify(new_X)
    print(new_feats.dim())
    return new_feats