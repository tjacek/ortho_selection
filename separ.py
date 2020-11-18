import numpy as np
import feats,files

class Clusters(object):
	def __init__(self, dataset,cats):
		self.dataset=dataset
		self.cats = cats

	def quality(self):
		centroids=self.get_centroids()
		vectors=self.get_vectors()
		for i,centroid_i in enumerate(centroids):
			in_i=vectors[i]
			out_i=[ vector_j
					for j,vector_j in enumerate(vectors)
						if(i!=j)]
			out_i=files.flatten(out_i)
			in_distance=np.mean(distance(centroid_i,in_i))
			out_distance=np.mean(distance(centroid_i,out_i))
			print(i)
			print(in_distance/out_distance)

	def get_centroids(self):
		centroids=[]
		for cat_i in self.get_vectors():
			X_i=np.array(cat_i)
			centroids.append(np.mean(X_i,axis=0))
		return centroids

	def get_vectors(self):
		return [[ self.dataset[name_j]
					for name_j in cat_i] 
						for cat_i in self.cats]

def distance(centroid_i,vectors):
	return [np.linalg.norm(centroid_i-vector_j) 
			for vector_j in vectors]

def make_clusters(dataset):
	cats=dataset.by_cat()
	dataset=dataset.to_dict()	
	return Clusters(dataset,cats)

def separ(in_path):
	dataset= feats.read(in_path)[0]
	train,test=dataset.split()
	clusters=make_clusters(train)
	clusters.quality()

separ("feats/stats/feats")