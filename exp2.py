import ens,files

class EnsEnsemble(object):
	def __init__(self,ensemble,gen=None):
		if(not gen):
			gen=GenPaths()
		self.ensemble=ensemble
		self.gen=gen

	def __call__(self,paths,out,binary=False,clf="LR",acc=True):
		for paths_i,out_i in self.gen(paths,out):
			print(paths_i)
			print(out_i)
			acc_i=self.ensemble(paths_i,clf=clf,
                    out_path=out_i,binary=binary,acc_only=acc)
			print(acc_i)

    def product_exp(self,paths,out,acc=True):
        args= [[True,False],["LR","SVC"]]
        arg_combs=files.iter_product(args) 
        for binary_i,clf_i in arg_combs:
            print(clf_i)
            self(paths,out,binary_i,clf_i,acc=acc)

class GenPaths(object):
	def __init__(self,common=None,binary=None):
		if(not common):
			common=["stats","basic","sim"]
		if(not binary):
			binary=["stats","basic","sim"]
		self.common=common
		self.binary=binary

	def __call__(self,paths,out):	
		old_common,old_deep=paths
		for feat_i in self.common:
			for feat_j in self.binary:
				out_i="%s_%s" % (feat_i,feat_j)
				common_i="%s/%s/feats" % (old_common,feat_i)
				deep_i="%s/%s/feats" % (old_deep,feat_j)
				new_paths=(common_i,deep_i)
				yield new_paths,out_i

in_dict="proj/MSR"
out_dict="MSR3"
common="scale"
deep_path="%s/ens" % in_dict

ens_exp=EnsEnsemble(None)
ens_exp((common,deep_path) ,"test")