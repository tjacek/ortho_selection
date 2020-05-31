from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support,accuracy_score

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