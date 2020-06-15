import files,ens,clf

clf_type="mixed"
feats=["stats","basic","sim"]
common_path="../smooth/common"
deep_path="../smooth/ens"
out="ens"
acc=True

def get_clf(raw_clf):
    if(raw_clf=="mixed"):
        return ["LR","SVC"]
    return raw_clf

def ens_exp(common_path,deep_path,feats,out,clf_type="LR",acc=True):
    ensemble=ens.get_ensemble()
    files.make_dir(out)
    files.make_dir("%s/%s" % (out,clf_type))
    for feat_i in feats:
        for feat_j in feats:
            out_ij="%s/%s/%s_%s" % (out,clf_type,feat_i,feat_j)
            hc_path="%s/%s/feats" % (common_path,feat_i)
            binary_path="%s/%s/feats" % (deep_path,feat_j)
            in_paths_i=(hc_path,binary_path)
            clf_ij=get_clf(clf_type)
            ensemble(in_paths_i,clf=clf_ij,out_path=out_ij,acc_only=acc)

ens_exp(common_path,deep_path,feats,out,clf_type,acc)