import numpy as np,random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#import tools,clf,learn.report,learn
import ens,learn

def make_curve(in_path,binary=False):
    votes=ens.get_votes(None,None,None,in_path)
    preds=[ learn.voting([vote_i],binary) 
            for vote_i in votes]
    y_true=votes[0][0]
    acc=[accuracy_score(y_true,pred_i) for pred_i in preds]
    print(acc)
#    show_curve(acc)

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
#random_ens(paths)
make_curve("votes/third")