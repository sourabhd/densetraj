from __future__ import print_function
import numpy as np
import h5py
from pyutils import Loader
from pyutils import SVCLearner
import time
import logging
import pdb
import sys
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def classify(tr, te, base_dir, dset_dir, cl):
    import pandas
    import time
    from pyutils import SVCLearner
    import numpy
    start = time.clock()
    labels_dict_file_train = dset_dir + '/' + 'ClipSets' + '/' + cl + '_' + 'train' + '.txt';
    labels_dict_file_test =  dset_dir + '/' + 'ClipSets' + '/' + cl + '_' + 'test' + '.txt';
    # print('%s\n' % labels_dict_file_train); 
    # print('%s\n' % labels_dict_file_test); 
    df_tr = pandas.read_csv(labels_dict_file_train, delim_whitespace=True, header=None)
    df_te = pandas.read_csv(labels_dict_file_test, delim_whitespace=True, header=None)
    # print(df_tr)
    # print(df_te)
    # print(numpy.atleast_2d(tr.data).shape)
    # print(numpy.atleast_2d(te.data).shape)
    # print(numpy.atleast_2d(df_tr.iloc[:,1]).shape)
    # print(numpy.atleast_2d(df_te.iloc[:,1]).shape)
    outfile = base_dir + '/' + 'results' + '/' + cl + '.txt'
    L = SVCLearner() 
    scores = L.learn(numpy.atleast_2d(tr.data),df_tr.iloc[:,1], \
            numpy.atleast_2d(te.data), df_te.iloc[:,1], df_te.iloc[:,0], outfile)
    precision, recall, thresholds = \
            precision_recall_curve(df_te.iloc[:,1], scores[:,0])
    average_precision = average_precision_score(df_te.iloc[:,1], scores[:,0])
    stop = time.clock()
    print('classify() for %s : %f sec\n' % (cl, stop-start))
    return (scores, precision, recall, thresholds, average_precision)


def main():
    dset_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Hollywood2';
    base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
    src_dir = '/nfs/bigeye/sdaptardar/actreg/densetraj';
    num_train = 823;
    num_test = 884;

    classes = [
    'AnswerPhone',
    'DriveCar',
    'Eat',
    'FightPerson',
    'GetOutCar',
    'HandShake',
    'HugPerson',
    'Kiss',
    'Run',
    'SitDown',
    'SitUp',
    'StandUp'
    ];

    num_classes = 12;

    feature_dir = base_dir + '/' + 'fisher';
    results_dir = base_dir + '/' + 'results'; 
    train_file = feature_dir + '/' + 'train_fv.mat';
    test_file = feature_dir + '/' + 'test_fv.mat';
    fileorder_f = src_dir + '/' + 'fileorder.mat';
    results_file = results_dir + '/' + 'classification.mat';

    # Load Fisher vectors

    feat_dim = 109056;
    tr_st = time.clock()
    tr = Loader(train_file, 'train_fv', num_train, feat_dim);
    tr.load_from_mat(sparse=False);
    tr_end = time.clock()
    print(tr.data)
    print(tr.data.shape)
    print("Fisher vectors loaded for train in %d sec" % (tr_end - tr_st))
    te_st = time.clock()
    te = Loader(test_file, 'test_fv', num_test, feat_dim);
    te.load_from_mat(sparse=False);
    te_end = time.clock()
    print(te.data)
    print(te.data.shape)
    print("Fisher vectors loaded for test in %d sec" % (te_end - te_st))

    scores = {}
    precision = {}
    recall = {}
    thresholds = {}
    average_precision = {}

    # Do the learning
    map_score = 0.0
    for cl in classes:
        scores[cl], precision[cl], recall[cl], thresholds[cl], \
        average_precision[cl] = classify(tr, te, base_dir, dset_dir, cl)
        print('| %20s | %10f |' % (cl, average_precision[cl]))
        map_score = map_score + average_precision[cl]
    map_score = map_score / num_classes
    print()
    print('| %20s | %10s | ' % ('Class', 'AP'))
    for cl in classes:
        print('| %20s | %10f |' % (cl, average_precision[cl]))
    print('MAP scores (ours) %f' % (map_score))

if __name__ == '__main__':
	t_start = time.clock()
	try:
		main()
		t_end = time.clock()
		print("Success")
		print("Time Taken: %.2f sec" % (t_end-t_start))
	except Exception, ex:
		t_end = time.clock()
		print("Failed")
		print("Time Taken: %.2f sec" % (t_end-t_start))
                logging.exception(ex)
		print("Attempting to drop to debugger")
		pdb.post_mortem(sys.exc_info()[2])


#te_f = load(test_file);
#fileorder = load(fileorder_f);

#mkdir(results_dir);
#
#model = cell(num_classes, 1);
#predicted_label = cell(num_classes, 1);
#accuracy = cell(num_classes, 1);
#probability_estimates = cell(num_classes, 1);
#recall = cell(num_classes, 1);
#precision = cell(num_classes, 1);
#ap_info = cell(num_classes, 1);
#test_fname = cell(num_classes, 1);
#test_true_labels = cell(num_classes, 1);
#decision_values = cell(num_classes, 1);
#pred_pos = cell(num_classes, 1);
#actual_pos = cell(num_classes, 1);
#
#for i = 1:num_classes
#%for i = 1:1
#    fprintf('%s\n', classes{i});
#    [model{i}, predicted_label{i}, accuracy{i}, decision_values{i}, probability_estimates{i}, ...
#     recall{i}, precision{i}, ap_info{i}, ...
#    test_fname{i}, test_true_labels{i}, ...
#    pred_pos{i}, actual_pos{i} ] = ...
#    run_classifier(dset_dir, base_dir, classes{i}, tr_f.train_fv, te_f.test_fv, fileorder);
#
#end
#
#save(results_file, 'base_dir', 'num_classes', 'classes', 'model',...
#     'predicted_label', 'accuracy', 'decision_values', 'probability_estimates', ...
#     'recall', 'precision', 'ap_info', 'pred_pos', 'actual_pos');
