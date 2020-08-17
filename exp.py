import files,ens,clf,inliners

clf_type="SVC"
feats=["stats","basic","sim"]
common_path="good/MHAD/common"
deep_path="good/MHAD/ens"
out="MHAD"
acc=True
binary=True

class EnsExp(object):
    def __init__(self,ensemble,prefix=False):
        self.ensemble=ensemble
        self.feats=["stats","basic","sim"]
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

def get_ensemble(inliner=False,prefix=True):
    ensemble= inliners.InlinerEnsemble() if(inliner) else ens.get_ensemble()
    return EnsExp(ensemble,prefix)

#def agum_exp(common_path,ens_path,out,clf_type="LR",acc=True):
#    ensemble=ens.get_ensemble()
#    feats=[None,"stats","basic"]
#    files.make_dir(out)
#    files.make_dir("%s/%s" % (out,clf_type))
#    for feat_i in feats:
#        for feat_j in [None,"basic"]:
#            hc_path=get_path(common_path,feat_i)
#            binary_path=get_path(ens_path,feat_j)
#            if(hc_path or binary_path):
#                out_ij="%s/%s/%s_%s" % (out,clf_type,feat_i,feat_j)
#                print(out_ij)
#                clf_ij=get_clf(clf_type)
#                in_paths_i=(hc_path,binary_path)
#                ensemble(in_paths_i,clf=clf_ij,out_path=out_ij,acc_only=acc)

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

ensemble=get_ensemble()
ensemble((common_path,deep_path),out,binary,clf_type,acc)
#agum_exp("../smooth/gap","../smooth/gap/ens","gap",clf_type="LR",acc=False)
#agum_exp("../smooth/common","../smooth/ens","raw",clf_type="LR",acc=False)