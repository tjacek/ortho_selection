import feats,files,tools
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np

def show_template(datasets,out_path,helpers):
    files.make_dir(out_path)
    for i,date_i in enumerate(datasets):
        type_i= helpers(i)
        out_i="%s/nn%d" % (out_path,i)
        plot_i=tsne_plot(date_i,show=False,color_helper=type_i)  
        plot_i.savefig(out_i,dpi=1000)
        plot_i.close()

def all_plots(common_path,deep_path,out_path=None,plot_type="cat"):
    datasets=tools.combined_dataset(common_path,deep_path)
    if(not out_path):
        out_path='plots_'+in_path
    if(plot_type=="single"):
        helper=lambda i:("single",i)
    else:
        helper=lambda i:plot_type
    show_template(datasets,out_path,helper)

def split_plot(in_path):
    dataset=feats.read(in_path)[0]
    def helper(i,y_i):
        name_i=dataset.info[i]
        person_i=int(name_i.split("_")[1])
        return (person_i % 2)
    tsne_plot(dataset,show=True,color_helper=helper)

def tsne_plot(in_path,show=True,color_helper="cat",names=False):
    feat_dataset= feats.read(in_path)[0] if(type(in_path)==str) else in_path
    feat_dataset=feat_dataset.split()[1]
    tsne=manifold.TSNE(n_components=2,perplexity=30)#init='pca', random_state=0)
    X=tsne.fit_transform(feat_dataset.X)
    y=feat_dataset.get_labels()
    names=feat_dataset.info if(names) else None
    if(type(color_helper)==str or type(color_helper)==tuple): 
        color_helper=get_colors_helper(feat_dataset.info,color_helper)
    return plot_embedding(X,y,title="tsne",color_helper=color_helper,show=show,names=names)

def plot_embedding(X,y,title="plot",color_helper=None,show=True,names=None):
    n_points=X.shape[0]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
   
    color_helper=color_helper if(color_helper) else lambda i,y_i:0

    plt.figure()
    ax = plt.subplot(111)

    rep= names if(names) else y
    for i in range(n_points):
        color_i= color_helper(i,y[i])
        plt.text(X[i, 0], X[i, 1],str(rep[i]),
                   color=plt.cm.tab20( color_i),
                   fontdict={'weight': 'bold', 'size': 9})
    print(x_min,x_max)
    if title is not None:
        plt.title(title)
    if(show):
        plt.show()
    return plt

def get_colors_helper(info,plot_type="cat"):
    index,fun=0,None
    if(plot_type=="full_person"):
        index=1
    if(plot_type=="person"):
        index,fun=1,lambda x:x%2
    if(type(plot_type)==tuple):
        index,cat_i=0,plot_type[1]
        fun=lambda x: int(x==(cat_i))
    def helper(i,y_i):
        desc=int(info[i].split('_')[index])
        if(fun):
            desc=fun(desc)
        return desc
    return helper    

if __name__ == "__main__":
    common_path=None
    deep_path='visual/sim'
#    all_plots(common_path,deep_path,"visual/plots","cat")
    split_plot('visual/sim/nn0')
