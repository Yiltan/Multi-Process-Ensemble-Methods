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
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

def logist_regression(categories):
	
	#print("\n---------- Logistic Regression -----------")
	
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
	
	#print("\n---------- Linear SVM -----------")
	
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
			#print(cnfsn_matrix)
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
	best_Ks = {}
	for category in categories:
		
		max_accuracies[category] = 0
		max_f1_scores[category] = 0
		best_Ks[category] = 0
		for k in k_range:
			if(f1_scores_list[k-1][category] > max_f1_scores[category]):
				best_Ks[category] = k
				max_f1_scores[category] = f1_scores_list[k-1][category]
				max_accuracies[category] = accuracies_list[k-1][category]
				cnfsn_matrices[category] = cnfsn_matrices_list[k-1][category]

	#print("KNN: best k for each category = ")
	#print(best_Ks)
	return max_f1_scores, max_accuracies, cnfsn_matrices, best_Ks


def gaussian_nb(categories):
	
	# print("\n---------- Gaussian -----------")
	
	f1_scores = {}
	accuracies = {}
	cnfsn_matrices = {}
	
	for category in categories:
		#print("\n------ Category: " + category)
		labels, data = dr.reduce_dimensions(dimensions=40, category=category)
	
		train_data, test_data, train_label, test_label = train_test_split(
			data, labels, test_size=0.15, random_state=0)

		kernel = 1.0 * RBF(1.0)
		
		classifier = GaussianProcessClassifier(kernel=kernel, random_state=0)
		classifier.fit(train_data, train_label)
		classifier_predicted = classifier.predict(test_data)
		
		f_one_score = f1_score(test_label, classifier_predicted, average='weighted')
		accuracy = metrics.accuracy_score(test_label, classifier_predicted)
		cnfsn_matrix = confusion_matrix(test_label, classifier_predicted)
		#print("f1_score: {0}".format(f_one_score))
		#print("Accuracy: {0}".format(accuracy))
		#print(cnfsn_matrix)
		#sys.stdout.flush()
		
		f1_scores[category] = f_one_score
		accuracies[category] = accuracy
		cnfsn_matrices[category] = cnfsn_matrix
	
	return f1_scores, accuracies, cnfsn_matrices
	
def ensemble(categories, KNN_best_Ks):
	
	# print("\n---------- Ensemble -----------")
	
	f1_scores = {}
	accuracies = {}
	cnfsn_matrices = {}
	
	for category in categories:
		#print("\n------ Category: " + category)
		labels, data = dr.reduce_dimensions(dimensions=40, category=category)
	
		train_data, test_data, train_label, test_label = train_test_split(
			data, labels, test_size=0.15, random_state=0)

		svc = svm.LinearSVC(dual=False, max_iter=10000)	
		knn = KNeighborsClassifier(n_neighbors=KNN_best_Ks[category])	
		logistic_regr = LogisticRegression(max_iter=10000)

		estimators = [('knn', knn), ('svc', svc), ('log_reg', logistic_regr)]
		ensemble = VotingClassifier(estimators, voting='hard', weights=[2,1,1])
		ensemble.fit(train_data, train_label)
		ensemble_predicted = ensemble.predict(test_data)
		
		f_one_score = f1_score(test_label, ensemble_predicted, average='weighted')
		accuracy = metrics.accuracy_score(test_label, ensemble_predicted)
		cnfsn_matrix = confusion_matrix(test_label, ensemble_predicted)
		#print("f1_score: {0}".format(f_one_score))
		#print("Accuracy: {0}".format(accuracy))
		#print(cnfsn_matrix)
		#sys.stdout.flush()
		
		f1_scores[category] = f_one_score
		accuracies[category] = accuracy
		cnfsn_matrices[category] = cnfsn_matrix
	
	return f1_scores, accuracies, cnfsn_matrices
# Used for testing
if __name__ == "__main__":
	categories = ['neutral','disgust','happiness','surprise','anger','fear','sadness']
	
	LR_f1, LR_accuracy, LR_cnfsn = logist_regression(categories)
	#GNB_f1, GNB_accuracy, GNB_cnfsn = gaussian_nb(categories)
	SVM_f1, SVM_accuracy, SVM_cnfsn = linear_svm(categories)
	KNN_f1, KNN_accuracy, KNN_cnfsn, KNN_best_Ks = knn(categories, range(1, 9)) #(1,25)
	ENS_f1, ENS_accuracy, ENS_cnfsn = ensemble(categories, KNN_best_Ks)
	
	# You can print the scores right away, like:
	# print(LR_f1)
	# print(LR_accuracy)
	
	for category in categories:
		print("\n------ Category: {0} -------".format(category))
		'''
		print("\n0. Gaussian Naive Bayes: ")
		print("\tf1_score: {0}".format(GNB_f1[category]))
		print("\tAccuracy: {0}".format(GNB_accuracy[category]))
		'''
		
		print("\n1. Logistic Regression: ")
		print("\tf1_score: {0}".format(LR_f1[category]))
		print("\tAccuracy: {0}".format(LR_accuracy[category]))
		
		print("\n2. SVM: ")
		print("\tf1_score: {0}".format(SVM_f1[category]))
		print("\tAccuracy: {0}".format(SVM_accuracy[category]))
		
		print("\n3. KNN: ")
		print("\tf1_score: {0}".format(KNN_f1[category]))
		print("\tAccuracy: {0}".format(KNN_accuracy[category]))
		print("\tBest K: {0}".format(KNN_best_Ks[category]))
		
		print("\n4. ENS: ")
		print("\tf1_score: {0}".format(ENS_f1[category]))
		print("\tAccuracy: {0}".format(ENS_accuracy[category]))
		
		sys.stdout.flush()
	