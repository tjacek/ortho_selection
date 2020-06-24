import ens,learn.report

def show_acc(paths):
    ensemble=ens.get_ensemble()
    datasets=[ (paths['common'],paths['ens']),(None,paths['ens']),(paths['common'],None)]
    acc=[]
    for data_i in datasets:
        acc_i=ensemble(data_i,clf="LR",acc_only=True)
        acc.append(acc_i)
    print("***************")    
    for acc_i in acc:
        print(acc_i)

def ens_stats(paths):
    result=learn.report.ens_acc((paths['common'],paths['ens']),clf="LR",acc_only=False)	
    print(learn.report.to_acc(result))

def get_paths(common_path,ens_path):
    return {'common':common_path,'ens':ens_path}

paths=get_paths("../outliners/common/stats/feats","../outliners/ens/sim/feats")
ens_stats(paths)