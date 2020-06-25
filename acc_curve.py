import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import tools,clf,learn.report,learn

def make_curve(in_path):
    y_true,results,names=get_votes(in_path)
    preds=[ learn.voting(results[:i],False) 
            for i in range(1,len(results))]
    acc=[accuracy_score(y_true,pred_i) for pred_i in preds]
    print(acc)
    show_curve(acc)

def get_votes(in_path):
    datasets=tools.read_datasets(in_path)
    acc=clf.cross_acc(datasets)
    ordering=np.flip(np.argsort(acc))
    result=learn.report.ens_acc(in_path,clf="LR",acc_only=False)
    y_true,names=result[0][0],result[0][2]
    results=[result[i] for i in ordering]
    return y_true,results,names

def show_curve(acc,name="acc_curve",out_path=None):
    plt.title(name)
    plt.grid(True)
    plt.xlabel('number of classifiers')
    plt.ylabel('accuracy')
    plt.plot(range(1,len(acc)+1), acc, color='red')
    if(out_path):
        plt.savefig(out_path)
    else:    
        plt.show()
    plt.clf()
    return acc

paths=("../outliners/common/stats/feats","../outliners/ens/sim/feats")
make_curve(paths)