clear all; close all; clc;
run('/nfs/bigeye/sdaptardar/installs/vlfeat/toolbox/vl_setup');
dbstop if error;

%base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/HollyWood2_BOF_Results';
base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
srcdir = '/nfs/bigeye/sdaptardar/actreg/densetraj';
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
fileorder_f = [ srcdir '/' 'fileorder.mat' ];

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

feat_dim = 109056;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAINING SET %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%  % x = dir(train_dir_glob);
%%   fileorder = load(fileorder_f);
%%   x = fileorder.train_files;
%%  % train_fv = zeros(num_train_dir, 218112);
%%  train_fv = zeros(num_train_dir, feat_dim);
%%   
%%   f = '';
%%   
%%   matlabpool close force 
%%   matlabpool open local 8 
%%   % for every file 
%%   %for i = 1:num_train_dir
%%   %parfor i = 1:10
%%   parfor i = 1:num_train_dir
%%       try
%%           time_start = tic;
%%           [~, hostname] = system('hostname','-echo');
%%           fprintf('Itr %d on %s\n', i, hostname);
%%           f = sprintf('%s%s%s', train_dir, '/', x(i).name);
%%           X = load(f); 
%%           D = cell(5,1);
%%           encoding = cell(5,1);
%%           sz = cell(5,1);
%%           data_dim = cell(5,1);
%%           red_data_dim = cell(5,1);
%%           % zdata = cell(5,1);
%%           cscores = cell(5,1);
%%           tdata = cell(5,1);
%%           mu_D = cell(5,1);
%%           sigma_D = cell(5,1);
%%           % for every feature
%%           for j = 1:5
%%               D{j} = X(:,DTF_desc_range_start(j):DTF_desc_range_end(j));
%%               sz{j} = size(D{j});
%%               data_dim{j} = sz{j}(2);
%%               red_data_dim{j} = data_dim{j} / 2;
%%               % zdata{j} = zscore(D{j});
%%               % cscores{j} = zdata{j} * G{j}.coefforth;
%%               mu_D{j} = repmat(G{j}.mu, size(D{j}, 1), 1);
%%               sigma_D{j} = repmat(G{j}.sigma, size(D{j}, 1), 1);
%%               cscores{j} = ((D{j} - mu_D{j}) ./ sigma_D{j}) * G{j}.coeff;
%%               tdata{j} = cscores{j}(:,1:red_data_dim{j});
%%               encoding{j} = vl_fisher(tdata{j}', G{j}.means, G{j}.covariances, G{j}.priors, 'Improved');
%%           end
%%           train_fv(i,:) =  vertcat(encoding{1:end})';
%%           time_elapsed = toc(time_start);
%%           fprintf('%f sec\n',  time_elapsed);
%%       catch ME
%%           disp(ME.message);
%%           disp(ME.stack);
%%           disp(ME.identifier);
%%           disp(ME);
%%           fprintf('Exception for %s', f);
%%       end
%%   end
%%   matlabpool close
%%   
%%   save(fisher_train_f, 'base_dir', 'train_fv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TEST SET %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 %x = dir(test_dir_glob);
 fileorder2 = load(fileorder_f);
 y = fileorder2.test_files;
 %test_fv = zeros(num_test_dir, 218112);
 test_fv = zeros(num_test_dir, feat_dim);
 
 f = '';
 
 matlabpool close force 
 matlabpool open local 8 
 % for every file 
 %for i = 1:num_test_dir
 parfor i = 1:num_test_dir
 %parfor i = 1:10
     try
         time_start = tic;
         [~, hostname] = system('hostname','-echo');
         fprintf('Itr %d on %s\n', i, hostname);
         f = sprintf('%s%s%s', test_dir, '/', y(i).name);
         X = load(f); 
         D = cell(5,1);
         encoding = cell(5,1);
         sz = cell(5,1);
         data_dim = cell(5,1);
         red_data_dim = cell(5,1);
         % zdata = cell(5,1);
         cscores = cell(5,1);
         tdata = cell(5,1);
         mu_D = cell(5,1);
         sigma_D = cell(5,1);
         % for every feature
         for j = 1:5
             D{j} = X(:,DTF_desc_range_start(j):DTF_desc_range_end(j));
             sz{j} = size(D{j});
             data_dim{j} = sz{j}(2);
             red_data_dim{j} = data_dim{j} / 2;
             % zdata{j} = zscore(D{j});
             % cscores{j} = zdata{j} * G{j}.coefforth;
             mu_D{j} = repmat(G{j}.mu, size(D{j}, 1), 1);
             sigma_D{j} = repmat(G{j}.sigma, size(D{j}, 1), 1);
             cscores{j} = ((D{j} - mu_D{j}) ./ sigma_D{j}) * G{j}.coeff;
             tdata{j} = cscores{j}(:,1:red_data_dim{j});
             encoding{j} = vl_fisher(tdata{j}', G{j}.means, G{j}.covariances, G{j}.priors, 'Improved');
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
