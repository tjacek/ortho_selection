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