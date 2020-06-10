import numpy as np
from sklearn.linear_model import LogisticRegression
#import learn

class BagEnsemble(object):
	def __init__(self,n_clf=3):
		self.clfs=[]
		self.n_clf=7

	def fit(self,X,y):
		for i in range(self.n_clf):
			clf_i=LogisticRegression(solver='liblinear')
			new_X,new_y=sample_data(X,y)
			clf_i.fit(new_X,new_y)
			self.clfs.append(clf_i)
		return self
    
	def predict(self,X):
		votes=[clf_i.predict(X) for clf_i in  self.clfs]

	def predict_proba(self,X):
		votes=[clf_i.predict_proba(X) for clf_i in  self.clfs]
		votes=np.sum(votes,axis=0)
#		votes=[np.argmax(vote_i) for vote_i in votes]
		return votes

def sample_data(X,y):
	sampled=np.random.randint(len(y),size=len(y))
	new_y=[y[i] for i in sampled]
	new_X=[X[i] for i in sampled]
	return new_X,new_y