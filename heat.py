import matplotlib.pyplot as plt
import seaborn
from sklearn.feature_selection import mutual_info_classif
import feats,files

def info_heatmap(in_path,out_path):
    data=feats.read_single(in_path,as_dict=False)
    info=[]
    for i in range(data.n_cats()):
        binary_i=data.binary(i)
        result=mutual_info_classif(binary_i.X,binary_i.get_labels())
        info.append(result)
    ax=seaborn.heatmap(info)
    ax.get_figure().savefig(out_path)
    plt.clf()

#info_heatmap("../proj2/basic/feats","test")
files.ens_template("../ens5/stats/feats","ens_stats",info_heatmap)