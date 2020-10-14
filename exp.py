import os.path#,itertools
import files,ens,clf,inliners
import learn.votes,tools

in_dict="proj/MSR"
out_dict="MSR3"
common="scale"

if(common=="mixed"):
    common_path=["%s/%s" % (in_dict,feat_i) 
                    for feat_i in ["agum","scale"]]
else:
    common_path= "%s/%s" % (in_dict,common) #["MSR_exp/agum","MSR_exp/scale"]
deep_path="%s/ens/" % in_dict
out="%s/%s" % (out_dict,common)
acc=True
inliner=False 

class EnsExp(object):
    def __init__(self,ensemble,prefix=False,common=None,
                    ens_feats=None):
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
                acc_ij=self.ensemble(tuple(new_paths),clf=clf_ij,
                    out_path=out_ij,binary=binary,acc_only=acc)
#                acc_ij=self.get_acc(new_paths,clf_ij,out_ij,binary,acc)
                if(self.prefix):
                    acc_ij="%s,%s,%s,%s,%s" % (feat_i,feat_j,clf_ij,str(binary),acc_ij)
                print(acc_ij)
    
    def product_exp(self,paths,out,acc=True):
        args= [[True,False],["LR"]]#,"SVC"]]
        arg_combs=files.iter_product(args) #list(itertools.product(*args))
        for binary_i,clf_i in arg_combs:
            print(clf_i)
            self(paths,out,binary_i,clf_i,acc=acc)

def get_paths(old_paths,common,deep):
    prefix_common,prefix_deep=old_paths
    deep_path="%s/%s/feats" % (prefix_deep,deep)
    if(type(common)==tuple):
        common_path=["%s/%s/feats" % (prefix_common,common_i) 
                        for common_i in common]
    else:
        common_path="%s/%s/feats" % (prefix_common,common)
    return common_path,deep_path
#def get_paths(old_paths,common,deep):
#    prefix_common,prefix_deep=old_paths
#    deep_path="%s/%s/feats" % (prefix_deep,deep)
#    if(type(prefix_common)==list):
#        common_path=["%s/%s/feats" % (common_i,common) 
#                for common_i in prefix_common]
#    else:
#        common_path="%s/%s/feats" % (prefix_common,common)
#    return common_path,deep_path

def get_ensemble(inliner=False,prefix=True,
                feats_type=False,concat=False):
    selection=tools.concat_dataset if(concat) else None
    ensemble= inliners.InlinerEnsemble() if(inliner) else ens.get_ensemble(selection)
    common,ens_feats=get_feat_types(feats_type)
    return EnsExp(ensemble,prefix,common,ens_feats)

def get_feat_types(feats_type):
    if(feats_type=="agum"):
        return ["stats","basic","sim"],["stats","basic","sim"]
    if(feats_type=="dtw"):
        return ["max_z"],["stats","basic","sim"]
    if(feats_type=="multi"):
        multi=[["max_z"],["stats","basic","sim"]]
        multi=files.iter_product(multi)
        return multi,["stats","basic"]
    return None,None

def get_clf(raw_clf):
    if(raw_clf=="mixed"):
        return ["LR","SVC"]
    return raw_clf                

def unify_votes(paths,out_path):
    paths=[files.top_files(path_i) 
            for path_i in paths ]
    paths=list(map(list, zip(*paths)))
    for path0,path1 in paths:
        out_i="%s/%s" % (out_path,path0.split("/")[-1])
        learn.votes.unify_votes([path0,path1],out_i)

def show_result(in_path,acc=True):
    paths=files.bottom_dict(in_path)
    for path_i in paths:
        for binary_i in [True,False]:
            acc_i=ens.read_result(path_i,binary_i,acc)
            print("%s,%s,%s" % (path_i,binary_i,acc_i))

paths=["MHAD/agum/SVC","MHAD/scale/SVC"]
out_path="MHAD/mixed/SVC"

show_result("MSR3/scale",True)
#unify_votes(paths,out_path)

ensemble=get_ensemble(inliner,feats_type="multi",concat=False)
ensemble.product_exp((common_path,deep_path),out,acc=acc)