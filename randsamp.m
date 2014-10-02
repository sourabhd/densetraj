
%base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/HollyWood2_BOF_Results';
base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
train_dir = [ base_dir '/'  'train'];
num_train_dir = 823;
train_dir_glob = [ train_dir '/' 'actioncliptrain*.txt' ];
num_feat_per_file = 320;
dim = 436;
M = zeros(num_train_dir * num_feat_per_file, dim);
sample_size = 256000;
random_sample_f = [ base_dir '/' 'feature_sample' '/' 'rand_sample_densetraj.mat' ];

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
        nr = min(Xsz(1), num_feat_per_file);
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

DTF = M(1:sample_size,:);

save(random_sample_f, 'base_dir', 'train_dir', 'num_train_dir', ...
            'sample_size', 'num_feat_per_file', 'dim', 'DTF');
fprintf('Completed\n');

