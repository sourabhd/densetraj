
run('/nfs/bigeye/sdaptardar/installs/vlfeat/toolbox/vl_setup');

base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/HollyWood2_BOF_Results';
train_dir = [ base_dir '/'  'train'];
test_dir = [ base_dir '/'  'test'];
feat_sample_dir = [ base_dir '/' 'feature_sample'];
gmm_dir = [ base_dir '/' 'gmm'];
fisher_dir = [ base_dir '/' 'fisher' ];
num_train_dir = 823;
num_test_dir = 884;
train_dir_glob = [ train_dir '/' 'actioncliptrain*.txt' ];
test_dir_glob = [ test_dir '/' 'actioncliptest*.txt' ];
num_feat_per_file = 320;
dim = 436;
sample_size = 256000;
random_sample_f = [ feat_sample_dir '/' 'rand_sample_densetraj.mat' ];
fisher_train_f = [ fisher_dir '/' 'train_fv.mat'  ];
fisher_test_f = [ fisher_dir '/' 'test_fv.mat'  ];

mkdir(fisher_dir)
DTF_desc_names = { 'Trajectory'; 'HOG'; 'HOF'; 'MBHx'; 'MBHy' };
DTF_desc_range_start = [11; 41; 137; 245; 341];
DTF_desc_range_end = [40; 136; 244; 340; 436];
DTF_desc_files = cell(5,1);
DTF_desc_fv = cell(5,1);
G = cell(5,1);

for i = 1:5 
    DTF_desc_files{i} = sprintf('%s%s%s%s', gmm_dir, '/', DTF_desc_names{i}, '.mat');
    G{i} = load(DTF_desc_files{i});
    size(G{i}.means)
    size(G{i}.covariances)
    size(G{i}.priors)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAINING SET %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% x = dir(train_dir_glob);
% train_fv = zeros(num_train_dir, 218112);
% 
% f = '';
% 
% matlabpool close force 
% matlabpool open local 8 
% for every file 
% parfor i = 1:num_train_dir
% %parfor i = 1:10
%     try
%         time_start = tic;
%         [~, hostname] = system('hostname','-echo');
%         fprintf('Itr %d on %s\n', i, hostname);
%         f = sprintf('%s%s%s', train_dir, '/', x(i).name);
%         X = load(f); 
%         D = cell(5,1);
%         encoding = cell(5,1);
%         for every feature
%         for j = 1:5
%             D{j} = X(:,DTF_desc_range_start(j):DTF_desc_range_end(j));
%             encoding{j} = vl_fisher(D{j}', G{j}.means, G{j}.covariances, G{j}.priors);
%         end
%         train_fv(i,:) =  vertcat(encoding{1:end})';
%         time_elapsed = toc(time_start);
%         fprintf('%f sec\n',  time_elapsed);
%     catch ME
%         disp(ME.message);
%         disp(ME.stack);
%         disp(ME.identifier);
%         disp(ME);
%         fprintf('Exception for %s', f);
%     end
% end
% matlabpool close
% 
% save(fisher_train_f, 'base_dir', 'train_fv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST SET %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x = dir(test_dir_glob);
test_fv = zeros(num_test_dir, 218112);

f = '';

matlabpool close force 
matlabpool open local 8 
% for every file 
parfor i = 1:num_test_dir
%parfor i = 1:10
    try
        time_start = tic;
        [~, hostname] = system('hostname','-echo');
        fprintf('Itr %d on %s\n', i, hostname);
        f = sprintf('%s%s%s', test_dir, '/', x(i).name);
        X = load(f); 
        D = cell(5,1);
        encoding = cell(5,1);
        % for every feature
        for j = 1:5
            D{j} = X(:,DTF_desc_range_start(j):DTF_desc_range_end(j));
            encoding{j} = vl_fisher(D{j}', G{j}.means, G{j}.covariances, G{j}.priors);
        end
        test_fv(i,:) =  vertcat(encoding{1:end})';
        time_elapsed = toc(time_start);
        fprintf('%f sec\n',  time_elapsed);
    catch ME
        disp(ME.message);
        disp(ME.stack);
        disp(ME.identifier);
        disp(ME);
        fprintf('Exception for %s', f);
    end
end
matlabpool close

save(fisher_test_f, 'base_dir', 'test_fv');
