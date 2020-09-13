import feats,reduction,tools
import inliners,inliners.knn

def visual(in_path):
    dataset=feats.read(in_path)[0]
    result=inliners.knn.get_detector(dataset,k=5,as_dict=True)	
    def helper(i,y_i):
        name_i=	dataset.info[i]
        person_i=int(name_i.split("_")[1])
        in_test= ((person_i % 2)==0)
        if(in_test):
            return result[name_i] + 1
        return in_test
    reduction.tsne_plot(dataset,show=True,color_helper=helper)

def show_inliners(paths,out_path):
    data=tools.combined_dataset(paths[0],paths[1],True) 
    full_data,deep_data=data[0],data[2]
    helpers=inliners.get_inliners(full_data,k=5)
    reduction.show_template(full_data,out_path,helpers)