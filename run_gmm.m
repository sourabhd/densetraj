function [means, covariances, priors, coeff, mean_X] = run_gmm(X, dataset, save_file)

run('/nfs/bigeye/sdaptardar/installs/vlfeat/toolbox/vl_setup');

fprintf('Running GMM for %s\n', dataset);

[N, D] = size(X); 
L = D / 2;
fprintf('N= %d D = %d L = %d\n', N, D, L);
start_time = tic; 
num_clusters = 256;
num_repetitions_kmeans = 20;
num_iter_kmeans = 1000;
num_repetitions_gmm = 1;
num_iter_gmm = 1000;

% Compute Principal components
pr_start = tic;
%mean_X = mean(X, 1);
[Z, mu, sigma]  = zscore(X);   % Standardize 
[coeff, score] = princomp(Z);
data = score(:,1:L)';
pr_time = toc(pr_start);
fprintf('Principal Components : time taken: %d\n', pr_time);


% K-means initialization
km_start = tic;
[initMeans, assignments] = vl_kmeans(data, num_clusters, ...
     'Algorithm','Elkan', ...
     'Initialization', 'plusplus', ...
     'NumRepetitions', num_repetitions_kmeans, ... 
     'MaxNumIterations', num_iter_kmeans, ...
     'verbose');
 
initCovariances = zeros(L,num_clusters);
initPriors = zeros(1,num_clusters);
 
 for i=1:num_clusters
     data_k = data(:,assignments==i);
     %initPriors(i) = size(data_k,2) / num_clusters;
     initPriors(i) = size(data_k,2) / size(data, 2);
     
     if size(data_k,1) == 0 || size(data_k,2) == 0
         initCovariances(:,i) = diag(cov(data'));
     else
         initCovariances(:,i) = diag(cov(data_k'));
     end
 end
 
 km_time = toc(km_start);
 fprintf('K means : time taken: %d\n', km_time);


% GMM 
gmm_start = tic;
 [means, covariances, priors] = vl_gmm(data, num_clusters, ...
                                 'Initialization','custom', ...
                                 'InitMeans',initMeans, ...
                                 'InitCovariances',initCovariances, ...
                                 'InitPriors',initPriors, ...
                                 'MaxNumIterations', num_iter_gmm, ...
                                 'NumRepetitions', num_repetitions_gmm, ...
                                 'verbose');

% 
% [means, covariances, priors] = vl_gmm(data, num_clusters, ...
%                                 'MaxNumIterations', num_iter_gmm, ...
%                                 'NumRepetitions', num_repetitions_gmm, ...
%                                 'verbose');

gmm_time = toc(gmm_start);
fprintf('GMM : time taken: %d\n', gmm_time);

gmm_time = toc(gmm_start);
fprintf('GMM : time taken: %d\n', gmm_time);

 save(save_file, 'dataset', ... 
                 'coeff', 'mu', 'sigma', 'score', ...
                 'num_clusters',  ... 
                 'initMeans', 'initCovariances', 'initPriors', ...
                 'means', 'covariances', 'priors');

% save(save_file, 'dataset', ... 
%                 'coeff', 'mean_X', 'score', ...
%                 'num_clusters', ... 
%                 'means', 'covariances', 'priors');

elapsed_time = toc(start_time);
fprintf('GMM func: time taken = %f sec', elapsed_time);

dbstop if naninf

end
