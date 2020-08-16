import numpy as np
import learn,files,feats

def make_votes(datasets,binary,clf):
    if(type(clf)==list):
        votes=[]
        for clf_i in clf:
            votes+=[learn.train_model(data_i,binary,clf_i) 
                        for data_i in datasets]
    else:
        votes=[learn.train_model(data_i,binary,clf) 
                    for data_i in datasets]
    if(binary):
        votes=[one_hot_result(vote_i) for vote_i in votes]
    return votes

def save_votes(votes,out_path):
    files.make_dir(out_path)
    for i,vote_i in enumerate(votes):
        out_i="%s/nn%d" % (out_path,i)
        data_i=feats.FeatureSet(vote_i[1],vote_i[2])
        data_i.save(out_i)

def read_votes(in_path):
    data=feats.read(in_path)
    votes=[]
    y_true=data[0].get_labels()
    for data_i in data:
        votes.append([y_true,data_i.X,data_i.info])
    return votes

def one_hot_result(result_i):
    return result_i[0],to_one_hot(result_i[1]),result_i[2]

def to_one_hot(y):
    n_cats=max(y)+1
    one_hot=[]    
    for y_i in y:
        vec_i=np.zeros(n_cats)
        vec_i[y_i]=1
        one_hot.append(vec_i)
    return np.array(one_hot)