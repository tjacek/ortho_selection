import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import feats,files

def info_histogram(in_path,out_path=None):
    data=feats.read(in_path)[0]
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
    data=feats.read(in_path)[0]
    files.make_dir(out_path)
    for i in range(data.n_cats()):
        binary_i=data.binary(i)
        out_i="%s/cat%d" % (out_path,i)
        result=mutual_info_classif(binary_i.X,binary_i.get_labels())
        make_plot(result,out_i)

cat_histogram("../proj2/stats/feats","cats1")	