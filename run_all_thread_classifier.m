%dbstop if error
clear all; close all; clc;
t_start = tic;

num_par_threads  = 4;

dset_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Hollywood2';
% base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/HollyWood2_BOF_Results';
base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
src_dir = '/nfs/bigeye/sdaptardar/actreg/densetraj';

classes = {
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
};

num_classes = 12;
retain_frac_threads = 1.0;

feature_dir = [ base_dir '/' 'fisher_thread' ];
results_dir = [ base_dir '/' 'results_thread' ]; 
train_file = [ feature_dir '/' 'train_fv.mat' ];
test_file = [ feature_dir '/' 'test_fv.mat' ];
tr_f = load(train_file);
te_f = load(test_file);
size(tr_f.train_fv)
size(te_f.test_fv)


feature_dir_video = [ base_dir '/' 'fisher' ];
train_file_video = [ feature_dir_video '/' 'train_fv.mat' ];
test_file_video = [ feature_dir_video '/' 'test_fv.mat' ];
tr_f_video = load(train_file_video);
te_f_video = load(test_file_video);
size(tr_f_video.train_fv)
size(te_f_video.test_fv)

fileorder_f = [ src_dir '/' 'fileorder.mat' ];
fileorder = load(fileorder_f);
retain_frac_threads_str = strrep(sprintf('%6.4f', retain_frac_threads), '.', '_');
results_file =  [ results_dir '/' 'classification__' retain_frac_threads_str '.mat'];


mkdir(results_dir);

model = cell(num_classes, 1);
predicted_label = cell(num_classes, 1);
accuracy = cell(num_classes, 1);
probability_estimates = cell(num_classes, 1);
recall = cell(num_classes, 1);
precision = cell(num_classes, 1);
ap_info = cell(num_classes, 1);
test_fname = cell(num_classes, 1);
test_true_labels = cell(num_classes, 1);
decision_values = cell(num_classes, 1);
pred_pos = cell(num_classes, 1);
actual_pos = cell(num_classes, 1);
loocvscore = cell(num_classes, 1);

myCluster = parcluster('local');
delete(myCluster.Jobs);
myPool = parpool(myCluster, num_par_threads);
parfor i = 1:num_classes
    fprintf('%s\n', classes{i});
    [model{i}, predicted_label{i}, accuracy{i}, decision_values{i}, probability_estimates{i}, ...
     recall{i}, precision{i}, ap_info{i}, ...
    test_fname{i}, test_true_labels{i}, ...
    pred_pos{i}, actual_pos{i}, ...
    loocvscore{i}] = ...
    run_thread_classifier(dset_dir, base_dir, classes{i}, tr_f, te_f, tr_f_video, te_f_video, fileorder, retain_frac_threads);
end
delete(myPool);
delete(myCluster.Jobs);

fprintf('\n');
mean_ap = 0;
for i = 1:num_classes
    fprintf('\n| %20s | %10f |',  classes{i}, ap_info{i}.ap);
    mean_ap = mean_ap + ap_info{i}.ap;
end
mean_ap = mean_ap / num_classes;
fprintf('\n');

fprintf('Mean AP : %f\n', mean_ap);

save(results_file, '-v7.3', 'base_dir', 'mean_ap', 'retain_frac_threads', 'num_classes', 'classes', 'model',...
     'predicted_label', 'accuracy', 'decision_values', 'probability_estimates', ...
     'recall', 'precision', 'ap_info', 'pred_pos', 'actual_pos', 'loocvscore');

t_elapsed = toc(t_start);
fprintf('\nTotal time for classification : %f sec\n', t_elapsed);
