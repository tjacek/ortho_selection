import learn,learn.report,feats
import selection,tools,clf,learn.votes,files
import numpy as np
import os.path

class Ensemble(object):
    def __init__(self,selection=None):
        self.selection=selection

    def __call__(self,in_path,binary=True,clf="LR",acc_only=False,
                    out_path=None,cf_path=None):
        result=self.get_result(in_path,binary,clf,out_path)
        return show_report(result,acc_only,out_path=cf_path)

    def get_result(self,in_path,binary=True,clf="LR",out_path=None):
        datasets=self.selection(in_path)
        votes=get_votes(datasets,binary,clf,out_path)
        y_true=votes[0][0]
        y_pred=learn.voting(votes,binary)
        return [y_true,y_pred,votes[0][2]]  

class CatEnsemble(object):
    def __init__(self,subsets=None):
        if(not subsets):
            subsets=[[1,2,4,5,9,12,17,19],
                     [0,3,6,7,8,10,13,11],
                     [5,13,14,15,16,17,18,19]]
        self.subsets=[set(subset_i) for subset_i in subsets]

    def __call__(self,in_path,binary=True,clf="LR",acc_only=False):
        dataset=tools.read_datasets(in_path)
        cat_datasets=[[self.cat_dataset(data_j,subset_i)
                        for data_j in dataset]
                            for subset_i in self.subsets]
        acc=[]
        for cat_i in cat_datasets:
            votes=get_votes(cat_i,binary,clf,None)
            y_true=votes[0][0]
            y_pred=learn.voting(votes,binary)
            result=[y_true,y_pred,votes[0][2]]
            acc_i=show_report(result,acc_only=acc_only)
            if(acc_only):
                acc.append(float(acc_i.split(",")[0]))
        if(acc_only):
            mean_acc=np.mean(acc)
            return acc,mean_acc

    def cat_dataset(self,dataset,subset_j):
        data_dict=dataset.to_dict()
        new_cats={ cat_i:i  for i,cat_i in enumerate(subset_j)}
        new_dict={}
        for name_i,value_i in data_dict.items():
            cat_i=int(name_i.split("_")[0])-1
            if(cat_i in subset_j):
                postfix="_".join(name_i.split("_")[1:])
                new_name_i="%d_%s" %(new_cats[cat_i]+1,postfix)
                print(new_name_i)
                new_dict[new_name_i]=value_i
        return feats.from_dict(new_dict)

def read_result(in_path,binary=False,acc=False):
    if(type(in_path)==list):
        votes=[learn.votes.read_votes(path_i)
            for path_i in in_path]
        votes=files.flatten(votes)
    else:
        votes=learn.votes.read_votes(in_path)
    if(binary):
        votes=[learn.votes.as_binary(vote_i) for vote_i in votes]
    y_pred=learn.voting(votes,binary)
    result=[votes[0][0], y_pred,votes[0][2]]
    if(acc=="raw"):
        return result
    return show_report( result,acc)

def all_exp(in1,in2,threshold=0.92):
    paths1=[files.top_files(path_i) for path_i in files.top_files(in1)]
    paths1=files.flatten(paths1)
    paths2=[files.top_files(path_i) for path_i in files.top_files(in2)]
    paths2=files.flatten(paths2)   
    for path_i in paths1:
        for path_j in paths2:
            acc_i=read_result([path_i,path_j],binary=False,acc=True)
            if(float(acc_i.split(",")[0])>threshold):
                format_result([path_i,path_j],acc_i)

def one_exp(in_path):
    paths=[files.top_files(path_i) 
            for path_i in files.top_files(in_path)]
    paths=files.flatten(paths)
    for path_i in paths:
        acc_i=read_result(path_i,binary=False,acc=True)        
        format_result(path_i,acc_i)

def format_result(path_i,acc_i):
    if(type(path_i)==list):
        paths=[path_j.split("/")[-2:] 
                for path_j in path_i]
        paths=files.flatten(paths)
    else:
        paths=path_i.split("/")[-2:]
    paths=",".join(paths)
    paths=paths.replace("_",",")
    print("%s,%s"%(paths,acc_i))

def get_votes(datasets,binary,clf,out_path):
    if(out_path and os.path.isdir(out_path)):
        votes=learn.votes.read_votes(out_path)
    else:
        votes=learn.votes.make_votes(datasets,False,clf)
    if(out_path):
        learn.votes.save_votes(votes,out_path)    
    if(binary):
        votes=[learn.votes.as_binary(vote_i) for vote_i in votes]
    return votes

def show_report(result,acc_only=False,out_path=None):
    if(acc_only):
        acc_i=learn.report.compute_score(result)
        return acc_i
    else:
        learn.report.show_result(result)
        learn.report.show_confusion(result,out_path=out_path)
 
def get_ensemble(selection=None):
    if(type(selection)==str):
        return Ensemble(clf.get_selection(selection))
    if(selection):
        pass
#        selection=selection_decorator(selection)
    else:
        selection=tools.read_datasets
    return Ensemble(selection)

def selection_decorator(selection):
	def selection_helper(in_path):
		datasets=tools.read_datasets(in_path)
		datasets=[selection(data_i) for data_i in datasets]
		return datasets
	return selection_helper

if __name__=="__main__":
#    ensemble=get_ensemble()
#    common_path="feats/stats/feats"
#    deep_path="feats/basic/"
#    paths=(common_path,deep_path)
#    acc_i=ensemble(paths,clf="LR",binary=True,acc_only=False)
#    print(acc_i)
    paths=["votes/maxz/LR/maxz_sim",
            "votes/base/SVC/stats_basic"]
#    read_result(paths[0],False)
    all_exp("votes/maxz","votes/base")
#    one_exp("votes/base")