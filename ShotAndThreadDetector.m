classdef ShotAndThreadDetector 

    properties (Constant)
        ffmpegBin = '/nfs/bigeye/sdaptardar/installs/ffmpeg-2.4.2-64bit-static/ffmpeg';
        baseDir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
        srcDir = '/nfs/bigeye/sdaptardar/actreg/densetraj';
        dsetDir = '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Hollywood2';
        framesDir = [ ShotAndThreadDetector.baseDir filesep 'frames' ];
        gifDir = [ ShotAndThreadDetector.baseDir filesep 'gif' ];
        gmmDir = [ ShotAndThreadDetector.baseDir filesep 'gmm' ];
        gmmDir2 = [ ShotAndThreadDetector.baseDir filesep 'gmm2' ];
        fisherDir2 = [ ShotAndThreadDetector.baseDir filesep 'fisher_thread' ];
        fileorderFile = [ShotAndThreadDetector.srcDir filesep 'fileorder.mat'];
        trainFile = [ShotAndThreadDetector.fisherDir2 filesep 'train_fv.mat'];
        testFile = [ShotAndThreadDetector.fisherDir2 filesep 'test_fv.mat'];
        flen = 109056;
        numThreads = 4;
    end

    methods 
        function obj = ShotAndThreadDetector()
            run('setup_shot_thread_detection');
            % dbstop if error;
            mkdir(ShotAndThreadDetector.framesDir);
            mkdir(ShotAndThreadDetector.gmmDir2);
            mkdir(ShotAndThreadDetector.fisherDir2);
        end
    end


    methods (Static)        

        function [F, threads_with_fv, threads, shots] = extractShotsThreads(video_num, video_fname, gmmModelFile)
            try 
                t_start = tic;

                % first extract frames
                t_ef_start = tic;
                clip.file = video_fname;
                [pathstr,name,ext] = fileparts(video_fname); 
                outFrmDir = [ ShotAndThreadDetector.framesDir filesep name ];
                mkdir(outFrmDir);
                opts.ffmpegBin = ShotAndThreadDetector.ffmpegBin;
                ML_VidClip.extFrms(clip, outFrmDir, opts);
                t_ef_elapsed = toc(t_ef_start);
                fprintf('P: %10d : %25s : %15s : %10f s\n', video_num, video_fname, 'extFrms', t_ef_elapsed);
                
                % get files in directory
                t_gf_start = tic;
                imFiles = ml_getFilesInDir(outFrmDir, 'png');
                t_gf_elapsed = toc(t_gf_start);
                fprintf('P: %10d : %25s : %15s : %10f s\n', video_num, video_fname, 'getFilesInDir', t_gf_elapsed);

                % extract video threads
                t_gt_start = tic;
                maxDist = 5;
                featType = 'sift';
                shldDisp = 1;
                [shotBnds, threads] = ML_VidThread.getThreads(imFiles, maxDist, featType, shldDisp);
                nFrm = length(imFiles);
                nShot = length(shotBnds) + 1;
                t_gt_elapsed = toc(t_gt_start);
                fprintf('P: %10d : %25s : %15s : %10f s\n', video_num, video_fname, 'getThreads', t_gt_elapsed);
                
                % display the extracted threads
    %            eventStart = 1;
    %            eventEnd = nFrm;
    %            ML_VidThread.dispThreads(length(imFiles), shotBnds, threads, eventStart, eventEnd);
    %            
    %            outHtmlDir = [ ShotAndThreadDetector.gifDir filesep name ];
    %            mkdir(outHtmlDir);
    %            ML_VidThread.createHtmls(imFiles, shotBnds, threads, outHtmlDir);
    %            fprintf('Use an Internet browser to open %s/shots.html to display shots\n', outHtmlDir);
    %            fprintf('Use an Internet browser to open %s/threads.html to display threads\n', outHtmlDir);

                shots = zeros(2, nShot);
                shots(1,1) = 1;
                shots(2, end) = nFrm;
                
                if ~isempty(shotBnds)
                    shots(1, 2:end)   = shotBnds;
                    shots(2, 1:end-1) = shotBnds-1;
                end

                disp(shots)
                celldisp(threads)

                % Fisher vector computations
                
                fv = cell(length(threads), 1);
                fv_vec = cell(length(threads), 1);
                threads_with_fv = {};
                for j = 1:length(threads)
                    t_fv_start = tic;
                    frmIdxss = cell(1, length(threads{j}));
                    for i=1:length(threads{j});
                        shotId = threads{j}(i);
                        frmIdxss{i} = shots(1,shotId):shots(2,shotId);                    
                    end            
                    frmIdxs = cat(2, frmIdxss{:}); % indexes of frames
                    disp(frmIdxs)
                    
                    % Get the Fisher Vector feature
                    try
                        %fprintf('%s\n', gmmModelFile);
                        fv{j} = ML_IDTD.fvEncode4dir(outFrmDir, 'png', frmIdxs, gmmModelFile);
                        fv_vec{j} = [fv{j}.trajXY ; fv{j}.trajHog ; fv{j}.trajHof ; fv{j}.trajMbhx ; fv{j}.trajMbhy ]';
                        % fv_vec{j} = [fv{j}.trajXY ; fv{j}.trajHog ; fv{j}.trajHof ; fv{j}.trajMbh ]'; % Sourabh
                        fprintf('\n')
                        fprintf('nnz trajXY  : %15d %15d\n', nnz(fv{j}.trajXY)  ,  prod(size(fv{j}.trajXY  )));
                        fprintf('nnz trajHog : %15d %15d\n', nnz(fv{j}.trajHog) ,  prod(size(fv{j}.trajHog )));
                        fprintf('nnz trajHof : %15d %15d\n', nnz(fv{j}.trajHof) ,  prod(size(fv{j}.trajHof )));
                        % fprintf('nnz trajMbh : %15d %15d\n', nnz(fv{j}.trajMbh),  prod(size(fv{j}.trajMbh)));
                        fprintf('nnz trajMbhx: %15d %15d\n', nnz(fv{j}.trajMbhx),  prod(size(fv{j}.trajMbhx)));
                        fprintf('nnz trajMbhy: %15d %15d\n', nnz(fv{j}.trajMbhy),  prod(size(fv{j}.trajMbhy)));
                        threads_with_fv{end+1} = j; 
                    catch
                        fv{j} = {};
                        fv_vec{j} = {};
                    end
                    fprintf('fv %d\n', j);
                    %disp(fv)

                    t_fv_elapsed = toc(t_fv_start);
                    fprintf('P: %10d : %25s : %5d : %15s : %10f s\n', video_num, video_fname, j, 'fvEncode4dir', t_fv_elapsed);
                end
                F = fv_vec;
                t_elapsed = toc(t_start);
                fprintf('Z: %10d : %25s : %10f  sec\n', video_num, video_fname, t_elapsed);
            catch err
                fprintf('extractShotsThreads: %s : %s\n', err.identifier, err.message);
            end
        end


        function extractShotsThreadsAll()
            
            try
                %gmmModelFile =  '/nfs/bigeye/sdaptardar/actreg/MyGradFuncs/Video/others/GMM.mat';
                gmmModelFile = [ ShotAndThreadDetector.gmmDir2 filesep 'GMM.mat']; 
                conv_gmm(ShotAndThreadDetector.gmmDir, gmmModelFile);

                L = load(ShotAndThreadDetector.fileorderFile);
                base_dir = ShotAndThreadDetector.baseDir;

%                tr_len = length(L.train_files);
%                tr_fv = cell(tr_len, 1);
%                tr_threads_with_fv = cell(tr_len, 1);
%                tr_threads = cell(tr_len, 1);
%                tr_shots = cell(tr_len, 1);
%                tr_name = cell(tr_len,1);
%                tr_fullname = cell(tr_len,1);
%                tr_t2r_hmap = {};
%
%                %matlabpool close force 
%                %matlabpool open local 8 
%                %parfor idx = 1:tr_len
%               % delete(gcp('nocreate'));
%               % delete(cluster.Jobs)'
%                %cl = parcluster;
%                %mypool = parpool(cluster, ShotAndThreadDetector.numThreads);
%                mypool = parpool(ShotAndThreadDetector.numThreads);
%                % for idx = 1:1
%                %parfor idx = 1:12
%                parfor idx = 1:tr_len
%                    [~,tr_name{idx},~] = fileparts(L.train_files(idx).name); 
%                    tr_fullname{idx} = [ ShotAndThreadDetector.dsetDir filesep 'AVIClips' filesep tr_name{idx} '.avi' ];
%                    fprintf('%s\n', tr_fullname{idx});
%                    [tr_fv{idx}, tr_threads_with_fv{idx}, tr_threads{idx}, tr_shots{idx}] = ShotAndThreadDetector.extractShotsThreads(idx, tr_fullname{idx}, gmmModelFile);    
%                end
%                delete(mypool);
%                %matlabpool close
%                train_fv = zeros(0,ShotAndThreadDetector.flen);
%                tr_cnt = 0;
%                % for i = 1:1
%                %for i = 1:12
%                for i = 1:tr_len
%                    for j = 1:length(tr_threads_with_fv{i})
%                        train_fv = [  train_fv ; tr_fv{i}{tr_threads_with_fv{i}{j}} ];
%                        tr_cnt = tr_cnt + 1;
%                        len = length(tr_t2r_hmap);
%                        if i > len
%                            tr_t2r_hmap{i} = {};
%                        end
%                        tr_t2r_hmap{i}{j} = tr_cnt;
%                        tr_r2t(tr_cnt).file = tr_name{i};
%                        tr_r2t(tr_cnt).fileNum = i;
%                        tr_r2t(tr_cnt).threadNum = tr_threads_with_fv{i}{j};
%                     end
%                end
%                save(ShotAndThreadDetector.trainFile, 'base_dir', 'train_fv', 'tr_fv', 'tr_threads_with_fv', 'tr_threads', 'tr_shots', 'tr_t2r_hmap', 'tr_r2t', 'tr_name', 'tr_fullname');

                te_len = length(L.test_files);
                te_fv = cell(te_len, 1);
                te_threads_with_fv = cell(te_len, 1);
                te_threads = cell(te_len, 1);
                te_shots = cell(te_len, 1);
                te_name = cell(te_len,1);
                te_fullname = cell(te_len,1);
                te_t2r_hmap = {};

                %matlabpool close force 
                %matlabpool open local 8 
                delete(gcp('nocreate'));
                cluster = parcluster('local');
                delete(cluster.Jobs)
                %mypool = parpool(cluster, ShotAndThreadDetector.numThreads);
                mypool = parpool(cluster,ShotAndThreadDetector.numThreads);
                parfor idx = 1:te_len
                %for idx = 1:12
                %for idx = 1:1
                    [~,te_name{idx},~] = fileparts(L.test_files(idx).name); 
                    te_fullname{idx} = [ ShotAndThreadDetector.dsetDir filesep 'AVIClips' filesep  te_name{idx} '.avi' ];
                    fprintf('%s\n', te_fullname{idx});
                    [te_fv{idx}, te_threads_with_fv{idx}, te_threads{idx}, te_shots{idx}]  = ShotAndThreadDetector.extractShotsThreads(idx, te_fullname{idx}, gmmModelFile);    
                end
                delete(mypool);

                %matlabpool close
                test_fv = zeros(0,ShotAndThreadDetector.flen);
                te_cnt = 0;
                % for i = 1:1
                %for i = 1:12
                for i = 1:te_len
                    for j = 1:length(te_threads_with_fv{i})
                        test_fv = [  test_fv ; te_fv{i}{te_threads_with_fv{i}{j}} ];
                        te_cnt = te_cnt + 1;
                        len = length(te_t2r_hmap);
                        if i > len
                            te_t2r_hmap{i} = {};
                        end
                        te_t2r_hmap{i}{j} = te_cnt;
                        te_r2t(te_cnt).file = te_name{i};
                        te_r2t(te_cnt).fileNum = i;
                        te_r2t(te_cnt).threadNum = te_threads_with_fv{i}{j};
                     end
                end
                save(ShotAndThreadDetector.testFile, 'base_dir', 'test_fv', 'te_fv', 'te_threads_with_fv', 'te_threads', 'te_shots', 'te_t2r_hmap', 'te_r2t', 'te_name', 'te_fullname');
            catch err
                fprintf('extractShotsThreadsAll: %s : %s\n', err.identifier, err.message);
            end
        end

    end
end
