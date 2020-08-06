from sklearn import svm
import ens

def inliner_ens(in_path):
    ensemble=ens.get_ensemble()
    result=ensemble.get_result(in_path)
    print(type(result))

#def ():

def get_one_svc(X_train):
    clf_i=svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf_i.fit(X_train)
    return clf_i

paths=("../outliners/common/stats/feats","../outliners/ens/sim/feats")
inliner_ens(paths)