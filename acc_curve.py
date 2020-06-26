import numpy as np,random
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

def random_ens(in_path,size=None,k=200,clf="LR"):
    results=learn.report.ens_acc(paths,clf,False)
    y_true=results[0][0]
    def helper(size):
        if(not size):
            size=random.randrange(len(results)-1)+1
        subset=random.sample(results, size)
        pred_i=learn.voting(subset,False)
        acc_i=accuracy_score(y_true,pred_i)
        return acc_i
    acc=np.array([helper(size) for i in range(k)])
    print("%.4f,%.4f,%.4f" % (np.mean(acc),np.median(acc),np.amax(acc)))

paths=("../outliners/common/stats/feats","../outliners/ens/sim/feats")
random_ens(paths)