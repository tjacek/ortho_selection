import itertools
import files,ens,clf,inliners

clf_type="LR"
feats=["stats","basic","sim"]
common_path=["third_exp/agum1","third_exp/simple"]#["exp_agum/agum/","exp_agum/scale/"]
deep_path="third_exp/ens/"
out="MSR/mixed"
acc=True
binary=False
inliner=False

class EnsExp(object):
    def __init__(self,ensemble,prefix=False,common=None,ens_feats=None):
        if(not common):
            common=["stats","basic","sim"]
        if(not ens_feats):
            ens_feats=["stats","basic","sim"]
        self.ensemble=ensemble
        self.common=common
        self.ens=ens_feats
        self.prefix=prefix

    def __call__(self,paths,out,binary=True,clf_type="LR",acc=True):
        files.make_dir(out)
        files.make_dir("%s/%s" % (out,clf_type))
        print(out)
        for feat_i in self.common:
            for feat_j in self.ens:
                out_ij="%s/%s/%s_%s" % (out,clf_type,feat_i,feat_j)
                new_paths=get_paths(paths,feat_i,feat_j)
                clf_ij=get_clf(clf_type)
                acc_ij=self.ensemble(tuple(new_paths),clf=clf_ij,out_path=out_ij,binary=binary,acc_only=acc)
                if(self.prefix):
                    acc_ij="%s,%s,%s,%s,%s" % (feat_i,feat_j,clf_ij,str(binary),acc_ij)
                print(acc_ij)

    def product_exp(self,paths,out,acc=True):
        args= [[True,False],["LR"]]#,"SVC"]]
        arg_combs= list(itertools.product(*args))
        for binary_i,clf_i in arg_combs:
            print(clf_i)
            self(paths,out,binary_i,clf_i,acc=acc)

def get_paths(old_paths,common,deep):
    prefix_common,prefix_deep=old_paths
    deep_path="%s/%s/feats" % (prefix_deep,deep)
    if(type(prefix_common)==list):
        common_path=["%s/%s/feats" % (common_i,common) 
                for common_i in prefix_common]
    else:
        common_path="%s/%s/feats" % (prefix_common,common)
    return common_path,deep_path


def get_ensemble(inliner=False,prefix=True,agum=False):
    ensemble= inliners.InlinerEnsemble() if(inliner) else ens.get_ensemble()
    if(agum):
        common=["stats","basic"]
        ens_feats=["stats","basic","sim"]
        return EnsExp(ensemble,prefix,common,ens_feats)
    return EnsExp(ensemble,prefix)

def get_clf(raw_clf):
    if(raw_clf=="mixed"):
        return ["LR","SVC"]
    return raw_clf                

ensemble=get_ensemble(inliner,agum=True)
#ensemble((common_path,deep_path),out,binary,clf_type,acc)
ensemble.product_exp((common_path,deep_path),out,acc=acc)
#agum_exp("../smooth/gap","../smooth/gap/ens","gap",clf_type="LR",acc=False)
#agum_exp("../smooth/common","../smooth/ens","raw",clf_type="LR",acc=False)