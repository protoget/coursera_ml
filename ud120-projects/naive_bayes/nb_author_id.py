#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
t0 = time()
cls = gnb.fit(features_train, labels_train)
training_time = round(time()-t0, 3)

t0 = time()
pred = [cls.predict(d) for d in features_test]
print "training time:", training_time, "s"
print "prediction time:", round(time()-t0, 3), "s"

from sklearn.metrics import accuracy_score
print "accuracy_score: %s" % accuracy_score(labels_test, pred)



#########################################################


