import feats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter

def simple_exp(data):
    y_true,y_pred,names=train_model(data)
    return accuracy_score(y_true,y_pred)

def train_model(data):
    if(type(data)==str):	
        data=feats.read(data)[0]
    data.norm()
    train,test=data.split()
    model=LogisticRegression(solver='liblinear')
    model.fit(train.X,train.get_labels())
    y_true=test.get_labels()
    y_pred=model.predict(test.X)
    return y_true,y_pred,data.info

def ensemble_exp(in_path):
    datasets=feats.read(in_path)
    results=[train_model(data_i) for data_i in datasets]
    y_true=results[0][0]
    votes=[result_i[1] for result_i in results]
    votes=list(map(list, zip(*votes)))
    y_pred=[Counter(vote_i).most_common(1)[0][0]
               for vote_i in votes]
    print(accuracy_score(y_true,y_pred))

if __name__=="__main__":
    ensemble_exp("../ens5/sim/feats")