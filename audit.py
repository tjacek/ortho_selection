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
    full_result=simple_exp(paths)
    cat_dict=learn.report.cat_by_error(full_result)
    print(learn.report.to_acc(result))
    print(cat_dict)
    for cat_i in cat_dict.keys():
        print(cat_i)
        acc_i=learn.report.cat_acc(result,cat_i)
        print(acc_i)

def get_paths(common_path,ens_path):
    return {'common':common_path,'ens':ens_path}

def simple_exp(paths):
    ensemble=ens.get_ensemble()
    path=(paths['common'],paths['ens'])
    return ensemble.get_result(path,clf="LR")

paths=get_paths("../outliners/common/stats/feats","../outliners/ens/sim/feats")
ens_stats(paths)