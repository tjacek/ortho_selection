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
        return combined_dataset(common_path,deep_path)
    return feats.read(in_path)

def combined_dataset(common_path,deep_path):
    if(not common_path):
        return feats.read(deep_path)
    if(not deep_path):
        return feats.read(common_path)
    common_data=feats.read(common_path)[0]
    deep_data=feats.read(deep_path)
    datasets=[common_data+ data_i 
                for data_i in deep_data]
    return datasets