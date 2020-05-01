import reader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#categories = ['neutral','disgust','happiness','surprise','anger','fear','sadness']

def reduce_dimensions(dimensions=40, category='neutral'):
	
	features = reader.get_features()
	
	all_labels = reader.get_all_assessments()
	
	# Split features from their user and video IDs
	user_video = features.loc[:, [0,1]].values
	features = features.drop([0, 1], axis=1)
	
	# Add current state of the user to the features, only one category
	features = pd.concat([features, all_labels[category]], axis=1)
	# print(features)
	
	features = StandardScaler().fit_transform(features)
	#features_DF = pd.DataFrame(data=features)
	
	pca = PCA(n_components=dimensions)
	principal_components = pca.fit_transform(features)
	principal_DF = pd.DataFrame(data=principal_components)
	
	#variances = np.var(principal_components, axis=0)
	#variances_ratio = variances / np.sum(variances)
	#print("Variance ratio of the features: ")
	#print(variances_ratio)
	#print("Sum of the variances ratio (out of 1): ")
	#print(np.sum(variances_ratio[:40]))
	
	
	final_PCs_DF = pd.concat([pd.DataFrame(user_video), principal_DF], axis=1)
	
	# select the category of the label:
	labels = all_labels[category+".1"]
	
	return labels, final_PCs_DF
	
# Used for testing
if __name__ == "__main__":
	
	# First send total number of dimensions and see what is the efficient number
	# of dimensions. It seems that with 40 dimensions, we hold 82% of total
	# variances.
	# reduce_dimensions(384)
	
	labels, final_PCs_DF = reduce_dimensions('happiness')
	print(labels)
	print(final_PCs_DF)