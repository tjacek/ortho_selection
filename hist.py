import numpy as np
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

def ts_plot(in_path,out_path):
    seqs={ files.clean_str(path_i):np.load(path_i) 
            for path_i in files.top_files(in_path)}
    files.make_dir(out_path)
    for name_i,seq_i in seqs.items():
        out_i="%s/%s" % (out_path,name_i)
        files.make_dir(out_i)
        for j,ts_j in enumerate(seq_i.T):
            out_ij="%s/%d" %(out_i,j)
            print(out_ij)
            fig = plt.figure()
#            plt.clf()
            ax = plt.axes()
            x = range(ts_j.shape[0])
            ax.plot(x,ts_j)
            plt.savefig(out_ij)
            plt.close()
#    print(list(seqs.values())[1].shape)

if __name__=="__main__":
    ts_plot("../skeleton/parsed","../skeleton/ts")