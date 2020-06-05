from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
import learn,tools

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

def get_acc(common_path,deep_path,clf="LR"):
    datasets=tools.combined_dataset(common_path,deep_path)
    return [train_model(data_i,True,clf,True) 
                for data_i in datasets]

def cat_acc(common_path,deep_path,cat_k,clf="LR"):
    datasets=tools.combined_dataset(common_path,deep_path)
    results= [learn.train_model(data_i,True,clf,False) 
                for data_i in datasets]
    y_true=results[0][0]
    cat_k_true=[ y_i==cat_k for y_i in y_true]
    acc=[]
    for result_i in results:
        pred_i=result_i[1]
        cat_pred_i=[ int(y_true[j]==pred_ij and cat_k_true[j]) 
                        for j,pred_ij in enumerate(pred_i)]
        acc.append(sum(cat_pred_i))
    cat_size=sum(cat_k_true)
    acc=[ acc_i/cat_size for acc_i in acc]
    return acc