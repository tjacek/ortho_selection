from sklearn.metrics import classification_report

def show_result(y_pred,y_true=None,names=None):
    if((not y_true) or (not names)):
        y_pred,y_true,names=y_pred
    print(classification_report(y_true, y_pred,digits=4))
    print(show_errors(y_pred,y_true,names))

def show_errors(y_pred,y_true,names):
    errors= [ pred_i!=true_i 
            for pred_i,true_i in zip(y_pred,y_true)]
    error_names=[ (names[i],y_pred[i])
                  for i,error_i in enumerate(errors)
                    if(error_i)]
    return error_names