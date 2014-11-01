t_start = tic;
% dbstop if error

num_threads = 13;

%addpath('/nfs/bigeye/sdaptardar/installs/matlab_sge');
addpath('/opt/matlab_r2014a/toolbox/distcomp/examples/integration/sge/shared');

cluster = parallel.cluster.Generic('JobStorageLocation', '/nfs/bigeye/sdaptardar/jobs');
set(cluster, 'HasSharedFilesystem', true);
set(cluster, 'ClusterMatlabRoot', '/opt/matlab_r2014a');
set(cluster, 'OperatingSystem', 'unix');
set(cluster, 'IndependentSubmitFcn', @independentSubmitFcn);
set(cluster, 'CommunicatingSubmitFcn', @communicatingSubmitFcn);
set(cluster, 'GetJobStateFcn', @getJobStateFcn);
set(cluster, 'DeleteJobFcn', @deleteJobFcn);
disp(cluster);

addpath('/nfs/bigeye/sdaptardar/actreg/densetraj');

mypool = parpool(cluster, num_threads);

keyboard

parfor i = 1:num_threads
    system('/nfs/bigeye/sdaptardar/actreg/densetraj/getcpuid');
end

delete(mypool);
delete(cluster);

%T = ShotAndThreadDetector();
%T.extractShotsThreadsAll(cluster);

t_elapsed = toc(t_start);
fprintf('Time taken: %f', t_elapsed);
