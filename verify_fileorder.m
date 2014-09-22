%%
% Uncomment to verify file order
% fileorder = load('fileorder.mat');
% for i = 1:num_classes-1
%   sum(strcmp(testing_labels_fname{i}, testing_labels_fname{i+1}))
%   sum(strcmp( [ cell2mat(testing_labels_fname{i}) repmat(['.txt'], num_te, 1) ]  , vertcat(fileorder.test_files(:).name)))
% end