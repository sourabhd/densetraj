%dbstop if error
clear all; close all; clc;
t_start = tic;

run_desc = 'Convert to LSSVM trials';
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

num_par_threads = 6;

dset_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Hollywood2';
base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
src_dir  = '/nfs/bigeye/sdaptardar/actreg/densetraj';

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

num_classes         = 12;
retain_frac_threads = 1.0;

feature_dir = [ base_dir '/' 'fisher_thread' ];
results_dir = [ base_dir '/' 'results_thread' ];
train_file  = [ feature_dir '/' 'train_fv.mat' ];
test_file   = [ feature_dir '/' 'test_fv.mat' ];
tr_f        = load(train_file);
te_f        = load(test_file);
size(tr_f.train_fv)
size(te_f.test_fv)


feature_dir_video = [ base_dir '/' 'fisher' ];
train_file_video  = [ feature_dir_video '/' 'train_fv.mat' ];
test_file_video   = [ feature_dir_video '/' 'test_fv.mat' ];
tr_f_video        = load(train_file_video);
te_f_video        = load(test_file_video);
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

model                 = cell(num_classes, 1);
predicted_label       = cell(num_classes, 1);
accuracy              = cell(num_classes, 1);
probability_estimates = cell(num_classes, 1);
recall                = cell(num_classes, 1);
precision             = cell(num_classes, 1);
ap_info               = cell(num_classes, 1);
test_fname            = cell(num_classes, 1);
test_true_labels      = cell(num_classes, 1);
decision_values       = cell(num_classes, 1);
pred_pos              = cell(num_classes, 1);
actual_pos            = cell(num_classes, 1);
loocvscore            = cell(num_classes, 1);

% Compute the kernel matrices
t_kernelcomp_start = tic;
Linear_K_video  = tr_f_video.train_fv * tr_f_video.train_fv';
Linear_KK_video = te_f_video.test_fv * tr_f_video.train_fv';
Linear_K        = tr_f.train_fv * tr_f.train_fv';
Linear_KK       = te_f.test_fv * tr_f.train_fv';
t_kernelcomp_elapsed = toc(t_kernelcomp_start);
fprintf('Kernel matrices computed in %f sec\n', t_kernelcomp_elapsed);

%myCluster = parcluster('local');
%delete(myCluster.Jobs);
%myPool = parpool(myCluster, num_par_threads);
%parfor i = 1:num_classes
for i = 1:num_classes
    fprintf('%s\n', classes{i});

    % Input params
    param(i).dset_dir            = dset_dir;
    param(i).base_dir            = base_dir;
    param(i).cl                  = classes{i};
    param(i).tr                  = tr_f;
    param(i).te                  = te_f;
    param(i).tr_v                = tr_f_video;
    param(i).te_v                = te_f_video;
    param(i).fileorder           = fileorder;
    param(i).retain_frac_threads = retain_frac_threads;
    param(i).Linear_K_video      = Linear_K_video;
    param(i).Linear_KK_video     = Linear_KK_video;
    param(i).Linear_K            = Linear_K;
    param(i).Linear_KK           = Linear_KK;
    param(i).subset_size_ub      = 2;

    % Call our function
    classifier(i) = Classifier(param(i));
    res(i) = classifier(i).classifyByThreadSelection();

end

%delete(myPool);
%delete(myCluster.Jobs);

mAP.lssvm_video_baseline = ...
    calc_mean_ap(classes, { res(:).lssvm_video_baseline });
mAP.lssvm_thread_baseline = ... 
    calc_mean_ap(classes, { res(:).lssvm_thread_baseline });

fprintf('\n');
fprintf('Results : Mean Average Precision\n');
fprintf('%-25s : %15f\n', 'Video baseline', mAP.lssvm_video_baseline);
fprintf('%-25s : %15f\n', 'Thread baseline', mAP.lssvm_thread_baseline);
fprintf('\n');


save(results_file, '-v7.3', 'base_dir', ...
    'retain_frac_threads', ...
    'num_classes', 'classes', 'res', 'mAP');

t_elapsed = toc(t_start);
fprintf('\nTotal time for classification : %f sec\n', t_elapsed);
