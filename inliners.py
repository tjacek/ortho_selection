from sklearn import svm
import ens,tools

def inliner_ens(paths):
    common_path,deep_path=paths
    result=tools.combined_dataset(common_path,deep_path,sub_datasets=True) 
    full_data,deep_data=result[0],result[2]
    detectors=get_one_svc(deep_data)
    print(len(detectors))
#    ensemble=ens.get_ensemble()
#    result=ensemble.get_result(in_path)
#    print(type(result))

def get_inliners_detector(deep_path):
    deep_datasets=tools.read_datasets(deep_path)
    return [get_one_svc(data_i) for data_i in deep_datasets]

def get_one_svc(data_i):
    if(type(data_i)==list):
        return [get_one_svc(data) for data in data_i]
    train,test=data_i.split()	
    clf_i=svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf_i.fit(train.X)
    return clf_i

paths=("../outliners/common/stats/feats","../outliners/ens/sim/feats")
inliner_ens(paths)