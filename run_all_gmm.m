
%base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/HollyWood2_BOF_Results';
base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
train_dir = [ base_dir '/'  'train'];
test_dir = [ base_dir '/'  'test'];
feat_sample_dir = [ base_dir '/' 'feature_sample'];
gmm_dir = [ base_dir '/' 'gmm'];
num_train_dir = 823;
train_dir_glob = [ train_dir '/' 'actioncliptrain*.txt' ];
num_feat_per_file = 320;
dim = 436;
sample_size = 256000;
random_sample_f = [ feat_sample_dir '/' 'rand_sample_densetraj_eqperclass.mat' ];


mkdir(gmm_dir)
DTF_desc_names = { 'Trajectory'; 'HOG'; 'HOF'; 'MBHx'; 'MBHy' };
DTF_desc_range_start = [11; 41; 137; 245; 341];
DTF_desc_range_end = [40; 136; 244; 340; 436];
DTF_desc_files = cell(5,1);
DTF_desc_fv = cell(5,1);

S = load(random_sample_f);

for i = 1:5 
    DTF_desc_files{i} = sprintf('%s%s%s%s', gmm_dir, '/', DTF_desc_names{i}, '.mat');
    DTF_desc_fv{i} = S.DTF(:, DTF_desc_range_start(i):DTF_desc_range_end(i));
end

failed = cell(5,1);
hostname = cell(5,1);

matlabpool close force 
matlabpool open 5

parfor i = 1:5
%for i = 1:5
    [failed{i},hostname{i}] = system('hostname');
    fprintf('Itr %d on %s\n', i, hostname{i});
    size(DTF_desc_fv{i})
    run_gmm(DTF_desc_fv{i}, DTF_desc_names{i}, DTF_desc_files{i});            
end

matlabpool close
