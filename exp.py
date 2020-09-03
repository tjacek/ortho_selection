import itertools
import files,ens,clf,inliners

clf_type="SVC"
feats=["stats","basic","sim"]
common_path="../simple/fore/agum1"
deep_path="../simple/fore/exp3/ens"
out="agum"
acc=True
binary=False
inliner=True

class EnsExp(object):
    def __init__(self,ensemble,prefix=False,feats=None):
        if(not feats):
            feats=["stats","basic","sim"]
        self.ensemble=ensemble
        self.feats=feats
        self.prefix=prefix

    def __call__(self,paths,out,binary=True,clf_type="LR",acc=True):
        files.make_dir(out)
        files.make_dir("%s/%s" % (out,clf_type))
        print(out)
        for feat_i in self.feats:
            for feat_j in self.feats:
                out_ij="%s/%s/%s_%s" % (out,clf_type,feat_i,feat_j)
                new_paths=[ "%s/%s/feats" % pair 
                    for pair in zip(paths,(feat_i,feat_j))]
                clf_ij=get_clf(clf_type)
                acc_ij=self.ensemble(tuple(new_paths),clf=clf_ij,out_path=out_ij,binary=binary,acc_only=acc)
                if(self.prefix):
                    acc_ij="%s,%s,%s,%s,%s" % (feat_i,feat_j,clf_ij,str(binary),acc_ij)
                print(acc_ij)

    def product_exp(self,paths,out):
        args= [[True,False],["LR","SVC"]]
        arg_combs= list(itertools.product(*args))
        for binary_i,clf_i in arg_combs:
            print(clf_i)
            self(paths,out,binary_i,clf_i,True)

def get_ensemble(inliner=False,prefix=True):
    ensemble= inliners.InlinerEnsemble() if(inliner) else ens.get_ensemble()
    return EnsExp(ensemble,prefix)

def get_agum_ensemble(inliner=False,prefix=True):
    ensemble= inliners.InlinerEnsemble() if(inliner) else ens.get_ensemble()
    return EnsExp(ensemble,prefix,feats=["stats","basic"])

def get_path(common_path,feat_i):
    if(feat_i):
        hc_path="%s/%s/feats" % (common_path,feat_i)
    else:
        hc_path=None
    return hc_path

def get_clf(raw_clf):
    if(raw_clf=="mixed"):
        return ["LR","SVC"]
    return raw_clf                

ensemble=get_agum_ensemble(inliner)
#ensemble((common_path,deep_path),out,binary,clf_type,acc)
ensemble.product_exp((common_path,deep_path),out)
#agum_exp("../smooth/gap","../smooth/gap/ens","gap",clf_type="LR",acc=False)
#agum_exp("../smooth/common","../smooth/ens","raw",clf_type="LR",acc=False)