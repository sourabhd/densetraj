%% Improved Dense Trajectory Feature Evaluation

close all; clear all; clc;
run('/nfs/bigeye/sdaptardar/installs/vlfeat/toolbox/vl_setup.m');

dset_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Hollywood2';
%base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/HollyWood2_BOF_Results';
base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
num_train_dir = 823;
num_test_dir = 884;

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

cvpr11_ap = [ 0.326, 0.88, 0.652, 0.814, 0.527, 0.296, 0.542, 0.658, 0.821, 0.625, 0.2, 0.652 ]';

feature_dir = [ base_dir '/' 'fisher' ];
results_dir = [ base_dir '/' 'results' ]; 
train_file = [ feature_dir '/' 'train_fv.mat' ];
test_file = [ feature_dir '/' 'test_fv.mat' ];

results_file = [ results_dir '/' 'classification.mat'];
results = load(results_file);

CM = cell(num_classes,1);
order = cell(num_classes,1);
testing_labels_fname = cell(num_classes,1);
testing_labels_vector = cell(num_classes, 1);
AP = zeros(num_classes, 1);

%% Confusion Matrices and Precision Recall Curves for all classes

fig = cell(num_classes, 1);
for i = 1:num_classes,
    %%
    cl = classes{i};
    disp(sprintf('%s', cl));
    labels_dict_file_test = sprintf('%s%s%s%s%s%s%s%s', dset_dir, '/', 'ClipSets', '/', cl, '_', 'test', '.txt');
    fprintf('%s\n', labels_dict_file_test); 
    [testing_labels_fname{i}, testing_labels_vector{i}] = textread(labels_dict_file_test, '%s %d');
    te_sz = size(testing_labels_fname{i});
    num_te = te_sz(1);
    
    te_ix = [ find(testing_labels_vector{i} == 1) ; find(testing_labels_vector{i} == -1)];
    
    %testing_labels_vector{i}(testing_labels_vector{i} == 1) = 1;
    %testing_labels_vector{i}(testing_labels_vector{i} == -1) = 0;
    
    [CM{i}, order{i} ] = confusionmat(testing_labels_vector{i}(te_ix,:), results.predicted_label{i});
    disp(sprintf('Confusion Matrix for %s\n', cl));
    disp(CM{i});
    disp(sprintf('\n'));
    AP(i) = results.ap_info{i}.ap;
    fig{i} = figure;
    %vl_pr(testing_labels_vector{i}, results.probability_estimates{i});
    
    vl_pr(testing_labels_vector{i}(te_ix,:), results.decision_values{i});
end


%% Class wise result summary
%

S = cell(num_classes, 1);
for i = 1:num_classes
    S{i} = sprintf('| %12s | %10f | %10f | %10d | %10d |', ...
        classes{i}, AP(i), cvpr11_ap(i), ...
        results.pred_pos{i}, results.actual_pos{i});
end

S = [ sprintf('| %12s | %10s | %10s | %10s | %10s |', ...
    'Classes', 'AP', 'CVPR11', 'PP', 'P') ; S ]

%% Mean Average Precision

disp(sprintf('Our    MAP = %f',  mean(AP)));

disp(sprintf('CVPR11 MAP = %f',  mean(cvpr11_ap)));

disp(sprintf('Improved trajectory paper MAP = %f', 64.3));
