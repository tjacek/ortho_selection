import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import feats

def info_histogram(in_path,out_path=None):
    data=feats.read(in_path)[0]
    result=mutual_info_classif(data.X,data.get_labels())
    plt.bar(range(data.dim()), result)
    if(out_path):
        plt.savefig(out_path)
    else:
        plt.show()

#def category_inf(data_i):


info_histogram("../proj2/basic/feats","test")	