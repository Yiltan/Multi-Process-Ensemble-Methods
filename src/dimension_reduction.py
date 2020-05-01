import reader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def reduce_dimensions(dimensions):
	
	features = reader.get_features()
	
	labels = reader.get_all_assessments()
	
	# Split features from their IDs
	user_video = features.loc[:, [0,1]].values
	features = features.drop([0, 1], axis=1)
	
	features = StandardScaler().fit_transform(features)
	featuresDF = pd.DataFrame(data=features)
	
	print(featuresDF.head())
	
	pca = PCA(n_components=dimensions)
	principalComponents = pca.fit_transform(features)
	principalDF = pd.DataFrame(data=principalComponents)
	print(principalDF.head())
	
	#variances = np.var(principalComponents, axis=0)
	#variances_ratio = variances / np.sum(variances)
	#print(variances_ratio)
	#print(np.sum(variances_ratio[:40]))
	
	finalDF = pd.concat([pd.DataFrame(user_video), principalDF], axis=1)
	finalDF_labeled = pd.concat([principalDF, labels], axis=1)
	
	print(finalDF.head())
	print(finalDF_labeled)
	
	
# Used for testing
if __name__ == "__main__":
	
	# First send total number of dimensions and see what is the efficient number
	# of dimensions. It seems that with 40 dimensions, we hold 82% of total
	# variances.
	# reduce_dimensions(384)
	
	reduce_dimensions(40)