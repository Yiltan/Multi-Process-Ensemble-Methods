import dimension_reduction as dr
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def logist_regression(categories):
	
	for category in categories:
		print("\n------ Category: " + category)
		labels, data = dr.reduce_dimensions(dimensions=40, category=category)
	
		train_data, test_data, train_label, test_label = train_test_split(
			data, labels, test_size=1/8.0, random_state=0)

		logistic_regr = LogisticRegression(max_iter=10000)
		
		logistic_regr.fit(train_data, train_label)
		
		predicted = logistic_regr.predict(test_data)
		
		f_one_score = f1_score(test_label, predicted, average='weighted')
		print("f1_score: {0}".format(f_one_score))
		print("Accuracy: {0}".format(logistic_regr.score(test_data, test_label)))
		
# Used for testing
if __name__ == "__main__":
	categories = ['neutral','disgust','happiness','surprise','anger','fear','sadness']
	
	logist_regression(categories)