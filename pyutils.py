"""
Loads .mat file and runs classifier on it
-----------------------------------------

References : 1. Matlab version 7.3 form is HDF5
             http://stackoverflow.com/questions/17316880/reading-v-7-3-mat-file-in-python

			 2. HDF5 file access in python
			 http://www.hdfgroup.org/HDF5/examples/api18-py.html

			 3. Creating sparse matrix in scipy from hdf5
			 http://pastebin.com/0KtJSX4w

"""

from __future__ import print_function
import scipy.io as sio
import sys
import pdb
from time import time
import logging
import h5py
from scipy.sparse import csc_matrix
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from copy import deepcopy

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

class Loader:

	def __init__(self, data_fname, data_vname, data_rows, data_cols):
		self.data_fname = data_fname
		self.data_vname = data_vname
		self.M = data_rows
		self.N = data_cols
		self.data = None

	def convert_to_sparse(self, h5file, vname):
		return csc_matrix((h5file[vname]['data'],h5file[vname]['ir'],h5file[vname]['jc']), shape=(self.M,self.N))

	def load_from_mat(self, sparse=False):
		try:
			print("Loading ... %s" % (self.data_fname))
			# Attempt normal .mat files
			self.data = sio.loadmat(self.data_fname)[self.data_vname]
		except Exception as ex:
			# if it v7.3 file : it is HDF5 format database
			f = h5py.File(self.data_fname)
			if sparse:
				self.data = self.convert_to_sparse(f,self.data_vname)
			else:
                                self.data = f[self.data_vname]
			f.close()
			#logging.exception("Failed to load matfile" % (self.data_fname))
		#print(self.data)


class Learner:
	def __init__(self, alldata, labels):
		data = deepcopy(alldata)
		print("00: (%d,%d)" %  (data.shape[0], data.shape[1]))
		imp = Imputer(missing_values='NaN', strategy='median', axis=0)
		imp.fit(data)
		self.data = deepcopy(imp.transform(data))
		print("0: (%d,%d)" %  (self.data.shape[0], self.data.shape[1]))
		le = LabelEncoder()
		le.fit(['f', 't'])
		self.labels = le.transform(labels)
	
	def fit_and_predict(self, train_idx, test_idx, sklearn_model, params):
		sklearn_model.params=params
		print("1: (%d,%d)" %  (self.data.shape[0], self.data.shape[1]))
		#print(self.data)
		sklearn_model.fit(self.data[train_idx,:],self.labels[train_idx])
		print("2: (%d,%d)" %  (self.data.shape[0], self.data.shape[1]))
		print(self.data.shape)
		print(type(self.data[1:2,:]))
		print(self.data[1:2,:].shape)
		print(self.data[train_idx,:])
		print("Heree .................................................")
		print(test_idx)
		print(self.data[test_idx,:])
		print("Heree .................................................")
		try:
			prob = sklearn_model.predict_proba(self.data[test_idx,:])
		except:
			scores = sklearn_model.decision_function(self.data[test_idx,:])
			prob = 1. / (1. + np.exp(-scores))
		return prob
			

class SVCLearner():
    """ Linear SVC Learning algorithm """
    def learn(self, X_train, y_train, X_test, y_test, id_test, outfile):
#        linearsvc_params = { 'penalty':'l2', 'loss':'l2', 'dual':True,
#                'tol':0.0001, 'C':1.0, 'multi_class':'ovr',
#                'fit_intercept':True, 'intercept_scaling':1,
#                'class_weight':'auto', 'verbose':2, 'random_state':None }
        clf = SVC(kernel='linear', class_weight='auto', C=100.0, verbose=2)
        clf.fit(X_train, y_train)
        scores = clf.decision_function(X_test)
        pred = clf.predict(X_test)
        ofile = open(outfile, 'w')
        for i in range(len(y_test)):
                ofile.write("%s,%f\n" % (id_test[i], scores[i]))
        ofile.close() 
        return (scores, pred)
