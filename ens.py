import learn,learn.report,feats
import selection,tools

class Ensemble(object):
    def __init__(self,selection=None):
    	if(selection):
            self.selection=selection_decorator(selection)
        else:
        	self.selection=read_datasets

    def __call__(self,in_path,binary=False,clf="LR",acc_only=False):
        datasets=self.selection(in_path)
        result=learn.ensemble_exp(datasets,
        	            binary=binary,clf=clf,acc_only=acc_only)
        if(acc_only):
            print(result)
        else:
            learn.report.show_result(result)
            learn.report.show_confusion(result)

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

ensemble=Ensemble(selection.basic_select)

paths=("../proj2/stats/feats","../ens5/basic/feats")
ensemble(paths)