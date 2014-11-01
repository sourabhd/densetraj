t_start = tic;
%dbstop if error

addpath('/nfs/bigeye/sdaptardar/actreg/densetraj');
T = ShotAndThreadDetector();
T.extractShotsThreadsAll();

t_elapsed = toc(t_start);
fprintf('Time taken: %f', t_elapsed);
