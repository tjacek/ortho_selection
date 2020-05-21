import feats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def simple_exp(data):
    if(type(data)==str):	
        data=feats.read(data)[0]
    data.norm()
    train,test=data.split()
    model=LogisticRegression(solver='liblinear')
    model.fit(train.X,train.get_labels())
    y_true=test.get_labels()
    y_pred=model.predict(test.X)
    print(y_pred)
    return accuracy_score(y_true,y_pred)

#exp("../proj2/stats/feats")