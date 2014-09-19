function [means, covariances, priors] = run_gmm(X, dataset, save_file)

run('/nfs/bigeye/sdaptardar/installs/vlfeat/toolbox/vl_setup');

fprintf('Running GMM for %s\n', dataset);

start_time = tic; 
data = X'; 
num_clusters = 256;
num_repetitions = 20;

[means, covariances, priors] = vl_gmm(data, num_clusters, ...
                                'NumRepetitions', num_repetitions, 'verbose');

save(save_file, 'dataset', ... 
                'num_clusters', 'num_repetitions', ... 
                'means', 'covariances', 'priors');
elapsed_time = toc(start_time);
fprintf('GMM: time taken = %f sec', elapsed_time);

end
