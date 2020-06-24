from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from collections import defaultdict
import learn,tools,files

def show_confusion(result):
    cf_matrix=confusion_matrix(result[0],result[1])
    print(cf_matrix)

def show_result(y_true,y_pred=None,names=None):
    if((not y_pred) or (not names)):
        y_true,y_pred,names=y_true
    print(classification_report(y_true, y_pred,digits=4))
    print(show_errors( [y_true,y_pred,names]))

def show_errors(result):
	result=list(map(list, zip(*result)))
	return [ result_i for result_i in result
	            if(result_i[0]!=result_i[1])]

def compute_score(result,as_str=True):
    y_true,y_pred=result[0],result[1]
    precision,recall,f1,support=precision_recall_fscore_support(y_true,y_pred,average='weighted')
    accuracy=accuracy_score(y_true,y_pred)
    if(as_str):
        return "%0.4f,%0.4f,%0.4f,%0.4f" % (accuracy,precision,recall,f1)
    else:
        return (accuracy,precision,recall,f1)

def ens_acc(paths,clf="LR",acc_only=True):
    if(type(paths)==tuple):
        common_path,deep_path=paths
        datasets=tools.combined_dataset(common_path,deep_path)
        return [learn.train_model(data_i,True,clf,acc_only) 
                    for data_i in datasets]
    votes=learn.read_votes(paths)
    y_true=[int(name_i.split("_")[0])-1 
                for name_i in votes[0][2]]
    result=[learn.voting([vote_i],False) for vote_i in votes]
    acc=[accuracy_score(y_true,result_i) for result_i in result]
    return acc
  
def to_acc(results):
    return [ accuracy_score(result_i[0],result_i[1]) for result_i in results]

def cat_acc(in_path,cat_k,clf="LR"):
    votes=[ learn.read_votes(path_i) 
            for path_i in files.top_files(in_path)]
    y_true=votes[0][0][0]
    preds=[learn.voting(vote_i,False) for vote_i in votes]
    acc=[ binary_acc(y_true,pred_i,cat_k) for pred_i in preds]
    return acc

def binary_acc(y_true,y_pred,cat_k):
    correct=[int(y_true[j]==pred_j and y_true[j]==cat_k) 
                for j,pred_j in enumerate(y_pred)]
    return sum(correct)/y_true.count(cat_k)

def cat_by_error(in_path):
    votes=learn.read_votes(in_path)
    y_true=votes[0][0]
    y_pred=learn.voting(votes,False)
    erorr=defaultdict(lambda:0)
    for true_i,pred_i in zip(y_true,y_pred):
        if(true_i!=pred_i):
            erorr[true_i]+=1
    norm=sum([value_j for value_j in erorr.values()])
    return { cat_j:(value_j/norm) 
                for cat_j,value_j in erorr.items()}