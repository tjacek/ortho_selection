import numpy as np
import tools,clf,learn.report

def show_curve(in_path):
    datasets=tools.read_datasets(in_path)
    acc=clf.cross_acc(datasets)
    print(acc)
    ordering=np.argsort(acc)
    ord_data=[datasets[i] for i in ordering]
    print(ordering)
    result=learn.report.ens_acc(in_path,clf="LR",acc_only=False)
    print(result[0][1].shape)

paths=("../outliners/common/stats/feats","../outliners/ens/sim/feats")
show_curve(paths)