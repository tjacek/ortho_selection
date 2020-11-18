import numpy as np
import feats

def separ(in_path):
	dataset= feats.read(in_path)[0]
	train,test=dataset.split()
	get_centroids(train)

def get_centroids(dataset):
	cats=dataset.by_cat()
	dataset=dataset.to_dict()
	centroids=[]
	for cat_i in cats:
		X_i=[dataset[name_j] 
				for name_j in cat_i]
		X_i=np.array(X_i)
		centroids.append(np.sum(X_i,axis=0))
	return centroids

separ("feats/stats/feats")