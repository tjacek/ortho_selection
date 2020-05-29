import learn,learn.report,feats
import selection

class Ensemble(object):
    def __init__(self,selection=None):
        self.selection=selection

    def __call__(self,in_path,binary=False,clf="LR",acc_only=False):
        datasets=self.read_datasets(in_path)
        if(self.selection):
            datasets=[self.selection(data_i) for data_i in datasets]
        result=learn.ensemble_exp(datasets,
        	            binary=binary,clf=clf,acc_only=acc_only)
        if(acc_only):
            print(result)
        else:
            learn.report.show_result(result)
            learn.report.show_confusion(result)

    def read_datasets(self,in_path):
        if(type(in_path)==tuple):
            common_path,deep_path=in_path	
            return learn.combined_dataset(common_path,deep_path)
        return feats.read(in_path)

ensemble=Ensemble(selection.basic_select)

paths=("../proj2/stats/feats","../ens5/basic/feats")
ensemble(paths)