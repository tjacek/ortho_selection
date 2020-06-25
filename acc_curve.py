import numpy as np
from sklearn.metrics import accuracy_score
import tools,clf,learn.report,learn

def show_curve(in_path):
    datasets=tools.read_datasets(in_path)
    y_true,results,names=get_votes(in_path,datasets)
    preds=[ learn.voting(results[:i],False) 
            for i in range(1,len(results))]
    acc=[accuracy_score(y_true,pred_i) for pred_i in preds]
    print(acc)

def get_votes(in_path,datasets):
    acc=clf.cross_acc(datasets)
    ordering=np.flip(np.argsort(acc))
#    ordering.reverse()
    result=learn.report.ens_acc(in_path,clf="LR",acc_only=False)
    y_true,names=result[0][0],result[0][2]
    results=[result[i] for i in ordering]
    return y_true,results,names

paths=("../outliners/common/stats/feats","../outliners/ens/sim/feats")
show_curve(paths)