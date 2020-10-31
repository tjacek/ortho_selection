import numpy as np
import feats,tools

def basic_select(data):
    if(type(data)==str):
        data=feats.read_single(data,as_dict=False)
    train,test=data.split()
    info=train.mutual()
    info=(info-np.mean(info))/np.std(info)
    return select_feats(data,info,lambda f: f>-1)

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
                if(cond(info[i]))]
    new_X=np.array(new_X).T
    new_feats=data.modify(new_X)
    print(new_feats.dim())
    return new_feats

def total_selection(in_path):
    common_path,deep_path=in_path   
    deep_data=feats.read(deep_path)
    deep_train=[deep_i.split()[0]
                    for deep_i in deep_data]
    info=[np.median(train_i.mutual())
            for train_i in deep_train]
    info=(info-np.mean(info))/np.std(info)
    print(info)
    common=feats.read(common_path)[0]
    datasets=[ common+data_i
                for i,data_i in enumerate(deep_data)
                    if(info[i]>-1)]
    return datasetst

def binary_selection(in_path):
    common_path,deep_path=in_path   
    deep_data=feats.read(deep_path)
    deep_data=[binary_helper(i,data_i) 
        for i,data_i in enumerate(deep_data)]
    common=feats.read(common_path)[0]
    datasets=[ common+deep_i for deep_i in deep_data]
    return datasets