import itertools
import feats

def filtered_dict(names,dic):
    return { name_i:dic[name_i] for name_i in names}

def split(names,selector=None):
    if(type(names)==dict):
        train,test=split(names.keys(),selector)
        return filtered_dict(train,names),filtered_dict(test,names)
    if(not selector):
        selector=get_person
    train,test=[],[]
    for name_i in names:
        if(selector(name_i)):
            train.append(name_i)
        else:
            test.append(name_i)
    return train,test

def get_person(name_i):
    return (int(name_i.split('_')[1])%2)==1

def person_cats(y):
    return ["%s_%d" %(y_i.split("_")[1],i) 
                for i,y_i in enumerate(y)]

def read_datasets(in_path):
    if(type(in_path)==tuple):
        common_path,deep_path=in_path
        if(type(common_path)==list):
            return multi_dataset(common_path,deep_path) 
        return combined_dataset(common_path,deep_path)
    return feats.read(in_path)

def combined_dataset(common_path,deep_path,sub_datasets=False):
    if(not common_path):
        return feats.read(deep_path)
    if(not deep_path):
        return feats.read(common_path)
    common_data=feats.read(common_path)[0]
    deep_data=feats.read(deep_path)
    datasets=[common_data+ data_i 
                for data_i in deep_data]
    if(sub_datasets):
        return datasets,common_data,deep_data
    return datasets

def multi_dataset(common_path,deep_path):
    datasets=[combined_dataset(common_i,deep_path)
                for common_i in common_path]
    return itertools.chain.from_iterable(datasets)

def concat_dataset(in_path):
    if(type(in_path)==tuple):
        common_path,deep_path=in_path
#        raise Exception(type(common_path))
        if(type(common_path)==list):
            common_data=feats.read_unified(common_path)
        else:
            common_data=feats.read(common_path)
#            return multi_dataset(common_path,deep_path) 
#        return combined_dataset(common_path,deep_path)
        deep_data=feats.read(deep_path)
        datasets=[common_data+ data_i 
                    for data_i in deep_data]
        return datasets
    return feats.read(in_path)