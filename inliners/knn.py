#from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors

def get_inliners(dataset,k=5):
    return [get_detector(data_i,k) for data_i in dataset]	

def get_detector(data_i,k=5):   
    train,test=data_i.split()	
    clf_i= neighbors.KNeighborsClassifier(k)
    clf_i.fit(train.X,train.get_labels())
    result=clf_i.predict(test.X)
    def helper(i,y_i):
        return int(result[i]==y_i)
    return helper