# -*- coding: utf-8 -*-
"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
import sklearn
from sklearn.tree import export_graphviz
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle

import labels


# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = '../data/all_labeled_data.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,2], data[i,3], data[i,4]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:2],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

data = np.nan_to_num(data)
#####################################################################################################
# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 20

# sampling rate should be about 100 Hz (sensor logger app)
n_samples = 1000
time_elapsed_seconds = (data[n_samples,1] - data[0,1])
sampling_rate = n_samples / time_elapsed_seconds

print("Sampling Rate: " + str(sampling_rate))

class_names = labels.activity_labels

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []
feature_names = []
for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,2:-1]
    # print("window = ")
    # print(window)
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
    
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------


#  split data into train and test datasets using 10-fold cross validation
cv = sklearn.model_selection.KFold(n_splits=10, random_state=None, shuffle=True)

#  calculate and print the average accuracy, precision and recall values over all 10 folds

total_accuracy = []
total_precision = []
total_recall = []
# X or Y or data or something like that
for (train_index, test_index) in cv.split(X,Y):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
    #
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    tree = sklearn.tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
    tree.fit(X_train, y_train) # fit tree on training set
    y_pred = tree.predict(X_test)
    conf = sklearn.metrics.confusion_matrix(y_test, y_pred, labels = tree.classes_) # confusion matrix

    # print out confusion matrix
    disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix = conf, display_labels = class_names)
    disp.plot()

    # print out values for each fold
    fold_accuracy = np.trace(conf) / float(np.sum(conf))
    total_accuracy.append(fold_accuracy)

    fold_precision = conf[1,1] / (conf[1,1] + conf[0,1])
    total_precision.append(fold_precision)

    fold_recall = conf[1,1] / (conf[1,1] + conf[1,0])
    total_recall.append(fold_recall)

# print the average accuracy, precision, and recall values
accuracy = np.mean(total_accuracy)
precision = np.mean(total_precision)
recall = np.mean(total_recall)

print("Avg Accuracy: " + str(accuracy) + ", Avg Precision: " + str(precision) + ", Avg Recall:" + str(recall) )

#  train the decision tree classifier on entire dataset
tree = sklearn.tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
# to fix nan temporarily
X = np.nan_to_num(X, nan=0.0)
tree.fit(X, Y)

## testing on demo data


#  Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line
export_graphviz(tree, out_file='tree.dot', feature_names = feature_names)

#  Save the classifier to disk - replace 'tree' with your decision tree and run the below line
print("saving classifier model...")
with open('classifier.pickle', 'wb') as f:
    pickle.dump(tree, f)
# %%
