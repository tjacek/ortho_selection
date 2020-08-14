import numpy as np
from sklearn import svm

def get_inliners(deep_data):
    detectors=get_detector(deep_data)
    return find_inliners(deep_data,detectors)

def find_inliners(deep_data,detectors):
    test_data=[ data_i.split()[1] for data_i in deep_data]
    inliners=[ detect_i.score_samples(test_i.X) 
                    for detect_i,test_i in zip(detectors,test_data)]
    inliners=np.array(inliners).T
    inliners[inliners>0]=1
    inliners=1-inliners
    return inliners

def get_detector(data_i):
    if(type(data_i)==list):
        return [get_detector(data) for data in data_i]
    train,test=data_i.split()	
    clf_i=svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf_i.fit_predict(train.X)
    return clf_i