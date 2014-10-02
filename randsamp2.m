
dbstop if error
clear all; close all; clc;

dset_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Hollywood2';
% base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/HollyWood2_BOF_Results';
base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
src_dir = '/nfs/bigeye/sdaptardar/actreg/densetraj';
num_train = 823;
num_test = 884;

classes = {      ...
'AnswerPhone',   ...
'DriveCar',      ...
'Eat',           ...
'FightPerson',   ...
'GetOutCar',     ...
'HandShake',     ...
'HugPerson',     ...
'Kiss',          ...
'Run',           ...
'SitDown',       ...
'SitUp',         ...
'StandUp'        ...
};

num_classes = 12;
num_pos = zeros(num_classes, 1);
label_mat = zeros(num_train, num_classes);

for i = 1:num_classes
%for i = 1:1

    cl = classes{i};
    labels_dict_file_train = sprintf('%s%s%s%s%s%s%s%s', dset_dir, '/', 'ClipSets', '/', cl, '_', 'train', '.txt');
    % fprintf('%s\n', labels_dict_file_train); 
    [training_labels_fname, training_labels_vector] = textread(labels_dict_file_train, '%s %d');
    tr_sz = size(training_labels_fname);
    num_tr = tr_sz(1);

    num_pos(i) = length(find(training_labels_vector == 1));
    num_samples = size(training_labels_vector, 1);

    label_mat(:,i) = training_labels_vector;

    fprintf('%20s %10d %10d \n', classes{i}, num_pos(i), num_samples);
end

label_mat(find(label_mat == -1)) = 0;

fprintf('Total Pos %d: \n', sum(num_pos));
sample_size = 256000;
samples_per_class = ceil(sample_size / num_classes);
%fprintf('Per class %d\n', samples_per_class);
%fprintf('Labels Matrix:\n');
samples_per_file = ceil(repmat(samples_per_class, 1, num_classes) ./ sum(label_mat, 1));
samples_from_file = sum(label_mat .* repmat(samples_per_file, num_train, 1), 2);
total_samples = sum(samples_from_file);
fprintf('Total samples %d\n', total_samples);


%base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/HollyWood2_BOF_Results';
base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
train_dir = [ base_dir '/'  'train'];
num_train_dir = 823;
train_dir_glob = [ train_dir '/' 'actioncliptrain*.txt' ];
num_feat_per_file = 320;
dim = 436;
%M = zeros(num_train_dir * num_feat_per_file, dim);
M = zeros(total_samples, dim);
sample_size = 256000;
random_sample_f = [ base_dir '/' 'feature_sample' '/' 'rand_sample_densetraj_eqperclass.mat' ];

% loop through the files once randomly picking up samples
x = dir(train_dir_glob);
nr = 0;
j = 1;
for i = 1:num_train_dir
% for i = 1:10
    try
        time_start = tic;
        fprintf('Itr %d : ', i)
        f = sprintf('%s%s%s', train_dir, '/', x(i).name);
        X = load(f);
        Xsz = size(X);
        nr = min(Xsz(1), samples_from_file(i));
        idx_r = randperm(Xsz(1))';
        idx = idx_r(1:nr,:);
        M(j:j+nr-1,:) = X(idx,:);
        j = j + nr;
        time_elapsed = toc(time_start);
        fprintf('%f sec\n',  time_elapsed);
    catch ME
        ME.message
        ME.stack
        ME.identifier
        ME
        fprintf('Exception for %s', f);
    end
end

%DTF = M(1:sample_size,:);
DTF = M;

save(random_sample_f, 'base_dir', 'train_dir', 'num_train_dir', ...
            'sample_size', 'num_feat_per_file', 'dim', 'DTF');
fprintf('Completed\n');

