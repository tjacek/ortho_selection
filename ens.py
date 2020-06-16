import learn,learn.report,feats
import selection,tools
import numpy as np
import os.path

class Ensemble(object):
    def __init__(self,selection=None):
        self.selection=selection

    def __call__(self,in_path,binary=False,clf="LR",acc_only=False,out_path=None):
        datasets=self.selection(in_path)
        if(out_path):
            if(os.path.isdir(out_path)):
                votes=learn.read_votes(out_path)
            else:
                votes=learn.make_votes(datasets,binary,clf)
                learn.save_votes(votes,out_path)
        else:
            votes=learn.make_votes(datasets,binary,clf)
        y_true=votes[0][0]
        y_pred=learn.voting(votes,binary)
        result=[y_true,y_pred,votes[0][2]]	
        if(acc_only):
             print(learn.report.compute_score(result))
        else:
            learn.report.show_result(result)
            learn.report.show_confusion(result)

def get_ensemble(selection=None):
    if(selection):
        selection=selection_decorator(selection)
    else:
        selection=read_datasets
    return Ensemble(selection)

def selection_decorator(selection):
	def selection_helper(in_path):
		datasets=read_datasets(in_path)
		datasets=[selection(data_i) for data_i in datasets]
		return datasets
	return selection_helper

def read_datasets(in_path):
    if(type(in_path)==tuple):
        common_path,deep_path=in_path	
        return tools.combined_dataset(common_path,deep_path)
    return feats.read(in_path)

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
    return datasets

if __name__=="__main__":
    ensemble=get_ensemble(None)#selection.complex_select)
#    paths=("../smooth/sim/feats","../ens5/basic/feats")
    paths=("../smooth/common/stats/feats","../smooth/ens/basic/feats")
#    ensemble(paths,clf="LR",out_path=None)
    acc=learn.report.cat_acc("raw/SVC",14)
    print(acc)