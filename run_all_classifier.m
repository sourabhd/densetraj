
dset_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Hollywood2';
base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/HollyWood2_BOF_Results';
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
}

num_classes = 12;

feature_dir = [ base_dir '/' 'fisher' ];
results_dir = [ base_dir '/' 'results' ]; 
train_file = [ feature_dir '/' 'train_fv.mat' ];
test_file = [ feature_dir '/' 'test_fv.mat' ];
tr_f = load(train_file);
te_f = load(test_file);
size(tr_f.train_fv)
size(te_f.test_fv)
fileorder = load('fileorder.mat');
results_file = [ results_dir '/' 'classification.mat'];

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

for i = 1:num_classes
%for i = 1:1
    fprintf('%s\n', classes{i});
    [model{i}, predicted_label{i}, accuracy{i}, probability_estimates{i}, ...
     recall{i}, precision{i}, ap_info{i}, ...
    test_fname{i}, test_true_labels{i} ] = ...
    run_classifier(dset_dir, base_dir, classes{i}, tr_f.train_fv, te_f.test_fv, fileorder);

end

save(results_file, 'base_dir', 'num_classes', 'classes', 'model',...
     'predicted_label', 'accuracy', 'probability_estimates', ...
     'recall', 'precision', 'ap_info');
