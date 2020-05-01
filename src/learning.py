import sys
import dimension_reduction as dr
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import sklearn.svm as svm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier


def logist_regression(categories):
	
	print("\n---------- Logistic Regression -----------")
	
	f1_scores = {}
	accuracies = {}
	cnfsn_matrices = {}
	
	for category in categories:
		#print("\n------ Category: " + category)
		labels, data = dr.reduce_dimensions(dimensions=40, category=category)
	
		train_data, test_data, train_label, test_label = train_test_split(
			data, labels, test_size=0.15, random_state=0)

		logistic_regr = LogisticRegression(max_iter=10000)
		logistic_regr.fit(train_data, train_label)
		predicted = logistic_regr.predict(test_data)
		
		f_one_score = f1_score(test_label, predicted, average='weighted')
		accuracy = metrics.accuracy_score(test_label, predicted)
		cnfsn_matrix = confusion_matrix(test_label, predicted)
		#print("f1_score: {0}".format(f_one_score))
		#print("Accuracy: {0}".format(accuracy))
		#print(cnfsn_matrix)
		#sys.stdout.flush()
		
		f1_scores[category] = f_one_score
		accuracies[category] = accuracy
		cnfsn_matrices[category] = cnfsn_matrix
	
	return f1_scores, accuracies, cnfsn_matrices

def linear_svm(categories):
	
	print("\n---------- Linear SVM -----------")
	
	f1_scores = {}
	accuracies = {}
	cnfsn_matrices = {}
	
	for category in categories:
		#print("\n------ Category: " + category)
		labels, data = dr.reduce_dimensions(dimensions=40, category=category)
	
		train_data, test_data, train_label, test_label = train_test_split(
			data, labels, test_size=0.15, random_state=0)

		svc = svm.LinearSVC(dual=False, max_iter=10000)
		svc.fit(train_data, train_label)
		predicted = svc.predict(test_data)
		
		f_one_score = f1_score(test_label, predicted, average='weighted')
		accuracy = metrics.accuracy_score(test_label, predicted)
		cnfsn_matrix = confusion_matrix(test_label, predicted)
		#print("f1_score: {0}".format(f_one_score))
		#print("Accuracy: {0}".format(accuracy))
		#print(cnfsn_matrix)
		#sys.stdout.flush()
		
		f1_scores[category] = f_one_score
		accuracies[category] = accuracy
		cnfsn_matrices[category] = cnfsn_matrix
	
	return f1_scores, accuracies, cnfsn_matrices

def knn(categories, k_range):

	accuracies_list = []
	f1_scores_list = []
	cnfsn_matrices_list = []
	for k in k_range:

		#print("---------- KNN-{0} -----------".format(k))
		accuracies = {}
		f1_scores = {}
		cnfsn_matrices = {}
		
		for category in categories:
			#print("\n------ Category: " + category)
			labels, data = dr.reduce_dimensions(dimensions=40, category=category)
		
			train_data, test_data, train_label, test_label = train_test_split(
				data, labels, test_size=0.15, random_state=0)

			knn = KNeighborsClassifier(n_neighbors=k)
			knn.fit(train_data, train_label)
			predicted = knn.predict(test_data)

			f_one_score = f1_score(test_label, predicted, average='weighted')
			accuracy = metrics.accuracy_score(test_label, predicted)
			cnfsn_matrix = confusion_matrix(test_label, predicted)
			#print("f1_score: {0}".format(f_one_score))
			#print("Accuracy: {0}".format(accuracy))
			# print(cnfsn_matrix)
			#sys.stdout.flush()
			
			f1_scores[category] = f_one_score
			accuracies[category] = accuracy
			cnfsn_matrices[category] = cnfsn_matrix
		
		accuracies_list.append(accuracies)
		f1_scores_list.append(f1_scores)
		cnfsn_matrices_list.append(cnfsn_matrices)
	
	# Select the best k based on the f1 score
	max_accuracies = {}
	max_f1_scores = {}
	cnfsn_matrices = {}
	best_k = {}
	for category in categories:
		
		max_accuracies[category] = 0
		max_f1_scores[category] = 0
		best_k[category] = 0
		for k in k_range:
			if(f1_scores_list[k-1][category] > max_f1_scores[category]):
				best_k[category] = k
				max_f1_scores[category] = f1_scores_list[k-1][category]
				max_accuracies[category] = accuracies_list[k-1][category]
				cnfsn_matrices[category] = cnfsn_matrices_list[k-1][category]

	print("KNN: best k for each category = ")
	print(best_k)
	return max_f1_scores, max_accuracies, cnfsn_matrices
		
# Used for testing
if __name__ == "__main__":
	categories = ['neutral','disgust','happiness','surprise','anger','fear','sadness']
	
	#LR_f1, LR_accuracy, LR_cnfsn = logist_regression(categories)
	#SVM_f1, SVM_accuracy, SVM_cnfsn = linear_svm(categories)
	KNN_f1, KNN_accuracy, KNN_cnfsn = knn(categories, range(1, 26))
	
	# You can print the scores right away, like:
	# print(LR_f1)
	# print(LR_accuracy)
	
	print(KNN_f1)
	
	