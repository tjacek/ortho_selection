import feats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def simple_exp(in_path):
    data=feats.read(in_path)[0]
    data.norm()
    train,test=data.split()
    model=LogisticRegression(solver='liblinear')
    model.fit(train.X,train.get_labels())
    y_true=test.get_labels()
    y_pred=model.predict(test.X)
    print(accuracy_score(y_true,y_pred))

exp("../proj2/stats/feats")