close all; clear all; clc;
run('/nfs/bigeye/sdaptardar/installs/vlfeat/toolbox/vl_setup.m');

dset_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Hollywood2';
base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
results_dir = [ base_dir '/' 'results' ]; 
py_results_file = [ results_dir '/' 'py_classification.mat'];
num_train = 823;
num_test = 884;
num_classes = 12;

py_res = load(py_results_file)

rc = cell(num_classes, 1);
pr = cell(num_classes, 1);
info = cell(num_classes, 1);
for i = 1:12
    [rc{i}, pr{i}, info{i}] = ... 
        vl_pr(py_res.true_test_labels(i,:), py_res.scores(i,:));
    fprintf('| %20s | %12f |\n', py_res.classes{i}, info{i}.ap);
end
