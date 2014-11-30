%dbstop if error
clear all; close all; clc;
t_start = tic;

run_desc = 'Convert to LSSVM trials: Train: avg of threeads ==> all but one, Test: avg of threads, Retain Fraction: 1.0, LOOCV criteria: leave on one video, validate on threads of left video, avg normalized, avoid recomputing the kernel';
fprintf('DESCRIPTION: %s\n', run_desc);

rng('shuffle');
rnd_out_prefix = char(randperm(26) + 96);

script_fullname = mfilename('fullpath');
[script_dir, script_name] = fileparts(script_fullname);


if isempty(script_dir)
    fprintf('SCRIPT NAME: %s\n', 'unknown');
else
    fprintf('SCRIPT NAME: %s\n', script_name);
    if strcmp(script_dir, '')
        fprintf('LAST GIT COMMIT: %s\n', 'unknown');
    else
        git_exec = '/usr/bin/git';
        git_rev_cmd = ...
            sprintf('%s --git-dir=%s%s%s rev-parse HEAD', ...
            git_exec, script_dir, filesep, '.git');
        [st, last_commit] = system(git_rev_cmd);
        fprintf('LAST GIT COMMIT: %s\n', last_commit);
    end
end

num_par_threads  = 6;

dset_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Hollywood2';
base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
src_dir = '/nfs/bigeye/sdaptardar/actreg/densetraj';

classes = {    ...
'AnswerPhone', ...
'DriveCar',    ...
'Eat',         ...
'FightPerson', ...
'GetOutCar',   ...
'HandShake',   ...
'HugPerson',   ...
'Kiss',        ...
'Run',         ...
'SitDown',     ...
'SitUp',       ...
'StandUp'      ...
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
retain_frac_threads_str = ... 
    strrep(sprintf('%6.4f', retain_frac_threads), '.', '_');
results_file = [ results_dir '/' rnd_out_prefix '__' 'classification__' ...
    retain_frac_threads_str '.mat'];

fprintf('RESULTS_FILE: %s\n', results_file);

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
Linear_K_video = tr_f_video.train_fv * tr_f_video.train_fv';
Linear_KK_video = te_f_video.test_fv * tr_f_video.train_fv';


%myCluster = parcluster('local');
%delete(myCluster.Jobs);
%myPool = parpool(myCluster, num_par_threads);
%parfor i = 1:num_classes
for i = 1:num_classes
    fprintf('%s\n', classes{i});

    % Input params
    param(i).dset_dir = dset_dir;
    param(i).base_dir = base_dir;
    param(i).cl = classes{i};
    param(i).tr = tr_f;
    param(i).te = te_f;
    param(i).tr_v = tr_f_video;
    param(i).te_v = te_f_video;
    param(i).fileorder = fileorder;
    param(i).retain_frac_threads = retain_frac_threads;
    param(i).Linear_K_video = Linear_K_video;
    param(i).Linear_KK_video = Linear_KK_video;

    % Call our function
    classifier(i) = Classifier(param(i));
    out(i) = classifier(i).run_thread_classifier_lssvm();

    % Get output parameters
%    model{i} = out(i).model;
%    predicted_label{i} = out(i).predicted_label;
%    accuracy{i} = out(i).accuracy;
%    decision_values{i} = out(i).decision_values;
%    probability_estimates{i} = out(i).probability_estimates;
    recall{i} = out(i).recall;
    precision{i} = out(i).precision;
    ap_info{i} = out(i).ap_info;
    test_fname{i} = out(i).testing_labels_fname;
%    test_true_labels{i} = out(i).testing_labels_vector;
%    pred_pos{i} = out(i).pred_pos;
%    actual_pos{i} = out(i).actual_pos;
%    loocvscore{i} = out(i).loocvscore;

    % Earlier code for function call 
%    [model{i}, predicted_label{i}, accuracy{i}, decision_values{i}, ...
%    probability_estimates{i}, ...
%    recall{i}, precision{i}, ap_info{i}, ...
%    test_fname{i}, test_true_labels{i}, ...
%    pred_pos{i}, actual_pos{i}, ...
%    loocvscore{i}] = ...
%    run_thread_classifier_lssvm(dset_dir, base_dir, ...
%        classes{i}, tr_f, te_f, tr_f_video, te_f_video, ...
%        fileorder, retain_frac_threads, Linear_K_video, Linear_KK_video);
%
end

%delete(myPool);
%delete(myCluster.Jobs);
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
