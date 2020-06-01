import files,ens,clf

clf_type="LR"
feats=["stats","basic","sim"]
common_path="../proj2"
deep_path="../ens5"
out="MSR3"
acc=True

def get_clf(raw_clf):
    if(raw_clf=="mixed"):
        return ["LR","SVC"]
    return raw_clf

ensemble=ens.Ensemble(clf.simple_selection)
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