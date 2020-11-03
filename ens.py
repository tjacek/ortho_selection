import learn,learn.report,feats
import selection,tools,clf,learn.votes,files
import numpy as np
import os.path

class Ensemble(object):
    def __init__(self,selection=None):
        self.selection=selection

    def __call__(self,in_path,binary=True,clf="LR",acc_only=False,out_path=None):
        result=self.get_result(in_path,binary,clf,out_path)
        return show_report(result,acc_only)

    def get_result(self,in_path,binary=True,clf="LR",out_path=None):
        datasets=self.selection(in_path)
        votes=get_votes(datasets,binary,clf,out_path)
        y_true=votes[0][0]
        y_pred=learn.voting(votes,binary)
        return [y_true,y_pred,votes[0][2]]  

def read_result(in_path,binary=False,acc=False):
    if(type(in_path)==list):
        votes=[learn.votes.read_votes(path_i)
            for path_i in in_path]
        votes=files.flatten_list(votes)
    else:
        votes=learn.votes.read_votes(in_path)
    if(binary):
        votes=[learn.votes.as_binary(vote_i) for vote_i in votes]
    y_pred=learn.voting(votes,binary)
    result=[votes[0][0], y_pred,votes[0][2]]
    if(acc=="raw"):
        return result
    return show_report( result,acc)

def all_exp(in1,in2):
    paths1=[files.top_files(path_i) for path_i in files.top_files(in1)]
    paths1=files.flatten_list(paths1)
    paths2=[files.top_files(path_i) for path_i in files.top_files(in2)]
    paths2=files.flatten_list(paths2)   
    for path_i in paths1:
        for path_j in paths2:
            print(path_i)
            print(path_j)
            print(read_result([path_i,path_j],binary=False,acc=True))

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

def show_report(result,acc_only=False):
    if(acc_only):
        acc_i=learn.report.compute_score(result)
        return acc_i
    else:
        learn.report.show_result(result)
        learn.report.show_confusion(result)
 
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
    ensemble=get_ensemble()#selection.complex_select)
#    common_path="proj/MSR/common/stats/feats"
#    deep_path=dir_path+"/ens/stats/feats"
#    paths=(None,deep_path)
#    acc_i=ensemble(paths,clf="LR",out_path=None,binary=False)
#    print(acc_i)

#    paths=["votes/maxz/LR/maxz_sim",
#            "votes/base/SVC/stats_basic"]
#    mixed_ensemble(paths,False)
#    all_exp("votes/maxz","votes/base")