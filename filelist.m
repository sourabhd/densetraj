
base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/HollyWood2_BOF_Results';
train_dir = [ base_dir '/'  'train'];
test_dir = [ base_dir '/'  'test'];
train_dir_glob = [ train_dir '/' 'actioncliptrain*.txt' ];
test_dir_glob = [ test_dir '/' 'actioncliptest*.txt' ];
train_files = dir(train_dir_glob);
test_files = dir(test_dir_glob);

save('fileorder.mat', 'train_files', 'test_files');
