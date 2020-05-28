from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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