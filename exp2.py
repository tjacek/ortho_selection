import ens,files,inliners

class ExpEnsemble(object):
	def __init__(self,ensemble,gen=None):
		if(not gen):
			gen=GenPaths()
		self.ensemble=ensemble
		self.gen=gen

	def __call__(self,paths,out,binary=False,clf="LR",acc=True,prefix=""):
		files.make_dir(out)
		out="%s/%s" % (out,clf)
		files.make_dir(out)		
		for paths_i,out_i in self.gen(paths):
			out_path="%s/%s" % (out,out_i)
			acc_i=self.ensemble(paths_i,clf=clf,out_path=out_path,binary=binary,acc_only=acc)
			result_i=self.format_result(out_path,prefix,acc_i)
			print(result_i)

	def format_result(self,out_path,prefix,acc_i):
		inliner_i=(type(self.ensemble)==inliners.InlinerEnsemble) 
		prefix_i=out_path.split('/')[-1]
		feat_desc=prefix_i.replace("_",",")
		tuple_i=(feat_desc,str(inliner_i),prefix,acc_i)
		result_i="%s,%s,%s,%s" % tuple_i
		return result_i

	def product_exp(self,paths,out,acc=True):
		args= [[True,False],["LR","SVC"]]
		arg_combs=files.iter_product(args) 
		for binary_i,clf_i in arg_combs:
			prefix="%s,%s" % (clf_i,str(binary_i))
			self(paths,out,binary_i,clf_i,acc=acc,prefix=prefix)

class GenPaths(object):
	def __init__(self,common=None,binary=None):
		if(not common):
			common=["stats","basic","sim"]
		if(not binary):
			binary=["stats","basic","sim"]
		self.common=common
		self.binary=binary

	def __call__(self,paths):	
		old_common,old_deep=paths
		for feat_i in self.common:
			for feat_j in self.binary:
				out_i="%s_%s" % (feat_i,feat_j)
				common_i=get_paths(common,feat_i)
				deep_i="%s/%s/feats" % (old_deep,feat_j)
				new_paths=(common_i,deep_i)
				yield new_paths,out_i

def get_paths(common,feat):
	if(type(feat)==list):
		common=[ "%s/%s/feats" % (common,feat_i)
				for feat_i in feat]
	else:
		common="%s/%s/feats" % (common,feat)
	return common

def get_exp_ens(inliner=False,gen=None):
	ensemble= inliners.InlinerEnsemble() if(inliner) else ens.get_ensemble()
	if(gen=="no_sim"):
		feats=["stats","basic"]
		gen=GenPaths(feats,feats)
	elif(gen=="dtw"):
		common=[["stats","max_z"],["basic","max_z"]]
		binary=["stats","basic","sim"]
		gen=GenPaths(common,binary)
	return ExpEnsemble(ensemble,gen)

def show_result(in_path,acc=True):
    paths=files.bottom_dict(in_path)
    result=[]
    for path_i in paths:
        for binary_i in [True,False]:
            acc_i=ens.read_result(path_i,binary_i,acc)
            prefix_i=path_i.split('/')[-1]
            clf_i=path_i.split('/')[-2]
            prefix_i=prefix_i.replace("_",",")
            tuple_i=(prefix_i,clf_i,binary_i,acc_i)
            str_i="%s,%s,%s,%s" % tuple_i
            print(str_i)
            result.append(str_i)
    return result

def get_common_feats(dataset,feat_type):
    if(feat_type=="mixed"):
        common=["exp/%s/%s" % (dataset,str_i) 
                for str_i in ["simple","agum"]]
    else:
        common="exp/%s/%s" % (dataset,feat_type)
    return common

if __name__ == '__main__':
#    dataset="proj"
#    feat_type="max_z3"
    out_path="votes/MSR2"#"%s/%s" % (dataset,feat_type)
    inliner=False
    common="MSR_exp/scale1"
    deep_path="MSR_exp/ens"
    ens_exp=get_exp_ens(inliner=inliner,gen="dtw")
    ens_exp.product_exp((common,deep_path),out_path,acc=True)
#    show_result(out_path,acc=True)