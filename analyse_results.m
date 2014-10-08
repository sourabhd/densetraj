%% Improved Dense Trajectory Feature Evaluation


close all; clear all; clc;
run('/nfs/bigeye/sdaptardar/installs/vlfeat/toolbox/vl_setup.m');
dbstop if error

dset_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Hollywood2';
dset_dir2 = '/tmp/Hollywood2/';
%base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/HollyWood2_BOF_Results';
base_dir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
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
};

num_classes = 12;
per_col = 10;
vid_width = 120;
vid_height = 90;


feature_dir = [ base_dir '/' 'fisher' ];
results_dir = [ base_dir '/' 'results' ]; 
train_file = [ feature_dir '/' 'train_fv.mat' ];
test_file = [ feature_dir '/' 'test_fv.mat' ];
html_dir = [ base_dir '/' 'html' ];
video_dir = [ dset_dir2 '/' 'Hollywood2' '/' 'AVIClips' ];

results_file = [ results_dir '/' 'classification.mat'];
results = load(results_file);

fname = cell(num_classes, 1);
sorted_decision_values = cell(num_classes, 1);
sorted_fname = cell(num_classes, 1);
s_ix = cell(num_classes, 1);
html = cell(num_classes, 1);
html_fname = cell(num_classes, 1);
mkdir(html_dir)


for i = 1:num_classes

    html_fname{i} = sprintf('%s%s%s%s', html_dir, '/', classes{i}, '.html');
    html{i} = fopen(html_fname{i}, 'w');
    fprintf(html{i}, '<!DOCTYPE html>');
    fprintf(html{i}, '<html>');
    fprintf(html{i}, '<head>');
    fprintf(html{i}, '<title>%s</title>', classes{i});
    fprintf(html{i}, '<base href="%s" />', video_dir);
    %fprintf(html{i}, '<embed type="application/x-vlc-plugin" pluginspage="http://www.videolan.org" />');
    %fprintf(html{i}, '<object classid="clsid:9BE31822-FDAD-461B-AD51-BE1D1C159921" codebase="http://download.videolan.org/pub/videolan/vlc/last/win32/axvlc.cab"></object>');
    fprintf(html{i}, '</head>');
    fprintf(html{i}, '<body>');

    cl = classes{i};
    labels_dict_file_test = sprintf('%s%s%s%s%s%s%s%s', dset_dir, '/', 'ClipSets', '/', cl, '_', 'test', '.txt');
    fprintf('%s\n', labels_dict_file_test); 
    [testing_labels_fname, testing_labels_vector] = textread(labels_dict_file_test, '%s %d');
    
    te_ix = [ find(testing_labels_vector == 1) ; find(testing_labels_vector == -1)];
    num_te = size(testing_labels_fname(te_ix,:), 1);
    fname{i} = [cell2mat(testing_labels_fname(te_ix,:)) repmat('.avi', num_te, 1) ]; 
    [sorted_decision_values{i}, s_ix{i}] = sort(results.decision_values{i});
    sorted_fname{i} = fname{i}(s_ix{i},:);
    fprintf(html{i}, '<table>')
    R = [repmat('<td><embed type="application/x-vlc-plugin" name="',num_te,1)  sorted_fname{i} repmat('" autoplay="no" width="120" height="90" target="',num_te, 1) sorted_fname{i} repmat('" /></td><td>', num_te, 1) num2str(sorted_decision_values{i}) repmat('</td>', num_te, 1)   ];
    disp(R)
    %RR = R(1:4,:)
    RR = [R(1:per_col,:) R(end-per_col+1:end,:)];
    RRR = [ repmat('<tr>', size(RR,1), 1) RR  repmat('</tr>', size(RR,1), 1) ]; 
    fprintf(html{i}, '%s', reshape(RRR',[],1));
    fprintf(html{i}, '</table>');
    fprintf(html{i}, '</body>\n</html>\n');
    fclose(html{i});

end
