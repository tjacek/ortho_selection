import numpy as np,random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import ens,learn,clf

def make_curve(in_path,binary=False,order=None,name="acc_curve",out_path=None,csv=True):
    acc=get_acc(in_path,binary,order)
    if(csv):
        np.save(out_path, acc)
    else:
        show_curve(acc,name=name,out_path=out_path)

def read_curve(in_path,name="acc_curve",out_path=None):
    acc=np.load(in_path)
    show_curve(acc,name=name,out_path=out_path)

def get_acc(in_path,binary=False,order=None):
    votes=ens.get_votes(None,binary,None,in_path)
    y_true=votes[0][0]
    if(not order is None):
        votes=[votes[k] for k in order]
    preds=[ learn.voting(votes[:i+1],binary) 
            for i,vote_i in enumerate(votes)]
    acc=[accuracy_score(y_true,pred_i) for pred_i in preds]
    print(acc)
    return acc

def show_curve(acc,name="acc_curve",out_path=None):
    if(not name):
        name="acc_curve"
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

def get_ordering(in_path):
    quality=clf.simple_quality(in_path)
    return np.argsort(quality)

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

#paths=(["feats/third/agum","feats/third/simple"],"feats/third/basic")
paths=("feats/MSR/stats","feats/MSR/basic")
#order= get_ordering(paths)
#order=np.flip(order)
#print(order)
#make_curve("selekcja/votes/MSR",binary=True,order=order,name="MSR",out_path="MSR")

read_curve("MSR.npy",name="acc_curve",out_path=None)