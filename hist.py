import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import feats,files

def info_histogram(in_path,out_path=None):
    data=feats.read_single(in_path,as_dict=False)
    result=mutual_info_classif(data.X,data.get_labels())
    make_plot(result,out_path)

def make_plot(result,out_path):
    plt.bar(range(result.shape[-1]), result)
    if(out_path):
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()

def cat_histogram(in_path,out_path):
    data=feats.read_single(in_path,as_dict=False)
    files.make_dir(out_path)
    for i in range(data.n_cats()):
        binary_i=data.binary(i)
        out_i="%s/cat%d" % (out_path,i)
        result=mutual_info_classif(binary_i.X,binary_i.get_labels())
        make_plot(result,out_i)

def ens_template(in_path,out_path,fun):
    files.make_dir(out_path)
    for path_i in files.top_files(in_path):
        out_i="%s/%s" %(out_path,path_i.split("/")[-1])
        fun(path_i,out_i)

ens_template("../ens5/basic/feats","ens_basic",cat_histogram)
