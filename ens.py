import learn,learn.report,feats
import selection,tools,clf,learn.votes
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
        selection=selection_decorator(selection)
    else:
        selection=tools.read_datasets
    return Ensemble(selection)

def selection_decorator(selection):
	def selection_helper(in_path):
		datasets=tools.read_datasets(in_path)
		datasets=[selection(data_i) for data_i in datasets]
		return datasets
	return selection_helper

def binary_selection(in_path):
    common_path,deep_path=in_path	
    deep_data=feats.read(deep_path)
    deep_data=[binary_helper(i,data_i) 
        for i,data_i in enumerate(deep_data)]
    common=feats.read(common_path)[0]
    datasets=[ common+deep_i for deep_i in deep_data]
    return datasets

def binary_helper(i,data_i):
    train_i=data_i.split()[0]
    train_i.info=tools.person_cats(train_i.info)
    binary_i=train_i#.binary(i)    
    info=binary_i.mutual()
#    print("%f,%f" % (np.mean(info),np.median(info)))
    info=(info-np.mean(info))/np.std(info)
    return selection.select_feats(data_i,
            info,lambda x:x<1)

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

if __name__=="__main__":
    ensemble=get_ensemble()#selection.complex_select)
    common_paths="exp_agum/agum/basic/feats"
    common_paths1="exp_agum/scale/basic/feats"
#    deep_path="../MSR_good/ens/sim/feats"
    deep_path="../../agum/outliners/ens/sim/feats"
    paths=([common_paths,common_paths1],deep_path)
    acc_i=ensemble(paths,clf="LR",out_path=None)
    print(acc_i)
