classdef VideoActionRecognizer < handle

    properties
        prop
    end

    methods 

        function ar = VideoActionRecognizer()
            run('/nfs/bigeye/sdaptardar/installs/vlfeat/toolbox/vl_setup');
            which vl_fisher
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
        function recognize(ar)
            t_start = tic;  
            ar.init();
            ar.mapTrThreadNum();
            ar.mapTeThreadNum();
            %ar.createCodebook();
            %ar.createLevel2TrainFV();
            %ar.createLevel2TestFV();
            ar.kernelComp();
            ar.createTrainSet();
            ar.createTestSet();
            %ar.kernelCompAug();
            ar.kernelCompVal();
            ar.kernelCompAvg();
            ar.classifyAllCategories();
            ar.presentResults();
            t_elapsed = toc(t_start);
            fprintf('\nTotal time for classification : %f sec\n', t_elapsed);
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
        function init(ar)

            ar.prop.num_par_threads = 6;
            ar.prop.run_desc = ...
                'LSSVM: train: pick best thread subset, test:video';
            fprintf('DESCRIPTION: %s\n', ar.prop.run_desc);

            rng('shuffle');
            rnd_out_prefix = char(randperm(26) + 96);
            script_fullname = mfilename('fullpath');
            [script_dir, script_name] = fileparts(script_fullname);

            if isempty(script_dir)
                fprintf('SCRIPT NAME: %s\n', 'unknown');
            else
                fprintf('SCRIPT NAME: %s\n', script_name);
                if strcmp(script_dir, '')
                    fprintf('LAST GIT COMMIT: %s\n', 'unknown');
                else
                    git_exec = '/usr/bin/git';
                    git_rev_cmd = ...
                        sprintf('%s --git-dir=%s%s%s rev-parse HEAD', ...
                        git_exec, script_dir, filesep, '.git');
                    [~, last_commit] = system(git_rev_cmd);
                    fprintf('LAST GIT COMMIT: %s\n', last_commit);
                end
            end

            ar.prop.dset_dir = ...
                '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Hollywood2';
            ar.prop.base_dir = ...
                '/nfs/bigeye/sdaptardar/Datasets/Hollywood2/Improved_Traj';
            ar.prop.src_dir  = '/nfs/bigeye/sdaptardar/actreg/densetraj';

            ar.prop.classes = {    ...
                'AnswerPhone', ...
                'DriveCar',    ...
                'Eat',         ...
                'FightPerson', ...
                'GetOutCar',   ...
                'HandShake',   ...
                'HugPerson',   ...
                'Kiss',        ...
                'Run',         ...
                'SitDown',     ...
                'SitUp',       ...
                'StandUp'      ...
                };

            ar.prop.num_classes         = 12;
            ar.prop.retain_frac_threads = 1.0;

            ar.prop.feature_dir = [ ar.prop.base_dir '/' 'fisher_thread' ];
            ar.prop.results_dir = [ ar.prop.base_dir '/' 'results_thread' ];
            ar.prop.train_file  = [ ar.prop.feature_dir '/' 'train_fv.mat' ];
            ar.prop.test_file   = [ ar.prop.feature_dir '/' 'test_fv.mat' ];
            ar.prop.tr_f        = load(ar.prop.train_file);
            ar.prop.te_f        = load(ar.prop.test_file);
            size(ar.prop.tr_f.train_fv)
            size(ar.prop.te_f.test_fv)
            ar.prop.num_tr = size(ar.prop.tr_f.train_fv,1);
            ar.prop.num_te = size(ar.prop.te_f.test_fv,1);
            ar.prop.feat_dim = size(ar.prop.tr_f.train_fv,2);

            % Normalize
            normalize = @(A) bsxfun(@times, A, 1 ./ sqrt(sum(A.^2,2)));
            ar.prop.tr_f.train_fv = normalize(ar.prop.tr_f.train_fv);
            ar.prop.te_f.test_fv = normalize(ar.prop.te_f.test_fv);

            mkdir(ar.prop.results_dir);

            ar.prop.feature_dir_video = [ ar.prop.base_dir '/' 'fisher' ];
            ar.prop.train_file_video  =  ...
                [ ar.prop.feature_dir_video '/' 'train_fv.mat' ];
            ar.prop.test_file_video   = ...
                [ ar.prop.feature_dir_video '/' 'test_fv.mat' ];
            ar.prop.tr_f_video        = load(ar.prop.train_file_video);
            ar.prop.te_f_video        = load(ar.prop.test_file_video);
            size(ar.prop.tr_f_video.train_fv)
            size(ar.prop.te_f_video.test_fv)
            ar.prop.num_tr_video = size(ar.prop.tr_f_video.train_fv,1);
            ar.prop.num_te_video = size(ar.prop.te_f_video.test_fv,1);

            % Normalize
            ar.prop.tr_f_video.train_fv = normalize(ar.prop.tr_f_video.train_fv);
            ar.prop.te_f_video.test_fv = normalize(ar.prop.te_f_video.test_fv);

            ar.prop.fileorder_f = [ ar.prop.src_dir '/' 'fileorder.mat' ];
            ar.prop.fileorder = load(ar.prop.fileorder_f);
            retain_frac_threads_str = ... 
                strrep(sprintf('%6.4f', ar.prop.retain_frac_threads), ...
                '.', '_');
            ar.prop.results_file = [ ar.prop.results_dir '/' rnd_out_prefix ...
                '__' 'classification__' ...
                retain_frac_threads_str '.mat'];

            fprintf('RESULTS_FILE: %s\n', ar.prop.results_file);

            ar.prop.subset_size_ub = 2;

        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
        function kernelComp(ar)
            % Compute the kernel matrices
            t_kernelcomp_start = tic;
            ar.prop.Linear_K_video  = ...
                ar.prop.tr_f_video.train_fv * ar.prop.tr_f_video.train_fv';
            ar.prop.Linear_KK_video = ...
                ar.prop.te_f_video.test_fv * ar.prop.tr_f_video.train_fv';
            ar.prop.Linear_K        = ...
                ar.prop.tr_f.train_fv * ar.prop.tr_f.train_fv';
            ar.prop.Linear_KK       = ...
                ar.prop.te_f.test_fv * ar.prop.tr_f.train_fv';
            t_kernelcomp_elapsed = toc(t_kernelcomp_start);
            fprintf('Kernel matrices computed in %f sec\n', ...
                t_kernelcomp_elapsed);
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function kernelCompAug(ar)
            % Compute the kernel matrices
            t_kernelcomp_start = tic;
            ar.prop.Linear_K_aug  = ...
                ar.prop.trX * ar.prop.trX';
            ar.prop.Linear_KK_aug = ...
                ar.prop.teX * ar.prop.trX';
            t_kernelcomp_elapsed = toc(t_kernelcomp_start);
            fprintf('Kernel matrices computed in %f sec\n', ...
                t_kernelcomp_elapsed);
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

        function kernelCompVal(ar)
            % Compute the kernel matrices
            t_kernelcomp_start = tic;
            ar.prop.Linear_K_val_avg_novideo = ...
                ar.prop.trX(1:ar.prop.trNumSubsets,:) * ar.prop.trXavg';
            ar.prop.Linear_K_val_avg_video = ...
                ar.prop.trX * ar.prop.trXavg';
            ar.prop.Linear_K_val_video_novideo = ...
                ar.prop.trX(1:ar.prop.trNumSubsets,:)*ar.prop.tr_f_video.train_fv';
            ar.prop.Linear_K_val_video_video = ...
                ar.prop.trX * ar.prop.tr_f_video.train_fv';
            t_kernelcomp_elapsed = toc(t_kernelcomp_start);
            fprintf('Kernel matrices computed in %f sec\n', ...
                t_kernelcomp_elapsed);
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
        function kernelCompAvg(ar)
            % Compute the kernel matrices
            t_kernelcomp_start = tic;
            ar.prop.Linear_K_avg = ...
                ar.prop.trXavg * ar.prop.trXavg';
            ar.prop.Linear_KK_avg = ...
                ar.prop.teXavg * ar.prop.trXavg';
            t_kernelcomp_elapsed = toc(t_kernelcomp_start);
            fprintf('Kernel matrices computed in %f sec\n', ...
                t_kernelcomp_elapsed);
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
        function classifyAllCategories(ar)

            for i = 1:ar.prop.num_classes
                fprintf('%s\n', ar.prop.classes{i});

                % Input params
                param(i).dset_dir             = ar.prop.dset_dir;
                param(i).base_dir             = ar.prop.base_dir;
                param(i).cl                   = ar.prop.classes{i};
                param(i).tr                   = ar.prop.tr_f;
                param(i).te                   = ar.prop.te_f;
                param(i).tr_v                 = ar.prop.tr_f_video;
                param(i).te_v                 = ar.prop.te_f_video;
                param(i).fileorder            = ar.prop.fileorder;
                param(i).retain_frac_threads  = ar.prop.retain_frac_threads;
                param(i).Linear_K_video       = ar.prop.Linear_K_video;
                param(i).Linear_KK_video      = ar.prop.Linear_KK_video;
                param(i).Linear_K             = ar.prop.Linear_K;
                param(i).Linear_KK            = ar.prop.Linear_KK;
                param(i).subset_size_ub       = ar.prop.subset_size_ub;
                param(i).trThread2rowMap      = ar.prop.trThread2rowMap;
                param(i).teThread2rowMap      = ar.prop.teThread2rowMap;
                param(i).trX                  = ar.prop.trX;
                param(i).teX                  = ar.prop.teX;
                param(i).numTrainSamplesVideo = ar.prop.numTrainSamplesVideo;
                param(i).numTestSamplesVideo  = ar.prop.numTestSamplesVideo;
                param(i).numTrainSamples      = ar.prop.numTrainSamples;
                param(i).numTestSamples       = ar.prop.numTestSamples;
                param(i).Linear_K_avg         = ar.prop.Linear_K_avg;
                param(i).Linear_KK_avg        = ar.prop.Linear_KK_avg;
%                param(i).Linear_K_aug         = ar.prop.Linear_K_aug;
%                param(i).Linear_KK_aug        = ar.prop.Linear_KK_aug;         
                param(i).Linear_K_val_video_video   = ar.prop.Linear_K_val_video_video;         
                param(i).Linear_K_val_video_novideo = ar.prop.Linear_K_val_video_novideo;         
                param(i).Linear_K_val_avg_video     = ar.prop.Linear_K_val_avg_video;         
                param(i).Linear_K_val_avg_novideo   = ar.prop.Linear_K_val_avg_novideo;         
                param(i).trXavg               = ar.prop.trXavg;
                param(i).teXavg               = ar.prop.teXavg;
                param(i).trNumSubsets         = ar.prop.trNumSubsets;
                param(i).teNumSubsets         = ar.prop.teNumSubsets;

                % Call our function
                classifier(i) = Classifier(param(i));
                ar.prop.res(i) = classifier(i).classifyByThreadSelection();

            end
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
        function presentResults(ar)

            fprintf('Video Baseline:\n')
            ar.prop.mAP.lssvm_video_baseline = ...
                calc_mean_ap(ar.prop.classes, ...
                { ar.prop.res(:).lssvm_video_baseline });
%            ar.prop.mAP.lssvm_thread_baseline = ... 
%                calc_mean_ap(ar.prop.classes, ...
%                { ar.prop.res(:).lssvm_thread_baseline });
%            ar.prop.mAP.lssvm_aug_baseline = ... 
%                calc_mean_ap(ar.prop.classes, ...
%                { ar.prop.res(:).lssvm_aug_baseline });
%
            fprintf('\n');
            fprintf('Thread Average Baseline:\n')
            ar.prop.mAP.lssvm_normavg_baseline = ... 
                calc_mean_ap(ar.prop.classes, ...
                { ar.prop.res(:).lssvm_normavg_baseline });

            fprintf('\n');
            fprintf('Start with: video ; include video with threads: YES');
            ar.prop.mAP.lssvmXS_video_video = ...
                calc_mean_ap(ar.prop.classes, ...
                { ar.prop.res(:).lssvmXS_video_video });

            fprintf('\n');
            fprintf('Start with: video ; include video with threads: NO');
            ar.prop.mAP.lssvmXS_video_novideo = ...
                calc_mean_ap(ar.prop.classes, ...
                { ar.prop.res(:).lssvmXS_video_novideo });

            fprintf('\n');
            fprintf('Start with: norm avg of threads ; include video with threads: YES');
            ar.prop.mAP.lssvmXS_avg_video = ...
                calc_mean_ap(ar.prop.classes, ...
                { ar.prop.res(:).lssvmXS_avg_video });


            fprintf('\n');
            fprintf('Start with: norm avg of threads ; include video with threads: NO');
            ar.prop.mAP.lssvmXS_avg_novideo = ...
                calc_mean_ap(ar.prop.classes, ...
                { ar.prop.res(:).lssvmXS_avg_novideo });


            fprintf('Results : Mean Average Precision\n');
            fprintf('%-25s : %15f\n', 'Video baseline', ...
                ar.prop.mAP.lssvm_video_baseline);
%            fprintf('%-25s : %15f\n', 'Thread baseline', ...
%                ar.prop.mAP.lssvm_thread_baseline);
%            fprintf('%-25s : %15f\n', 'Augmented baseline', ...
%                ar.prop.mAP.lssvm_aug_baseline);
            fprintf('%-25s : %15f\n', 'Our algorithm (video, video)', ...
                ar.prop.mAP.lssvmXS_video_video);
            fprintf('%-25s : %15f\n', 'Our algorithm (video, novideo)', ...
                ar.prop.mAP.lssvmXS_video_novideo);
            fprintf('%-25s : %15f\n', 'Our algorithm (avg, video)', ...
                ar.prop.mAP.lssvmXS_avg_video);
            fprintf('%-25s : %15f\n', 'Our algorithm (avg, novideo)', ...
                ar.prop.mAP.lssvmXS_avg_novideo);
            fprintf('\n');

%
%            save(ar.prop.results_file, 'ar.prop.base_dir', ...
%                'ar.prop.retain_frac_threads', ...
%                'ar.prop.num_classes', 'ar.prop.classes', ...
%                'ar.prop.res', 'ar.prop.mAP');

        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function mapTrThreadNum(ar)
            % Map thread numbers for lookup

            rCounter = 1;
            tKeySet = {};
            tValueSet = 1:ar.prop.num_tr;
            for i = 1:ar.prop.num_tr_video
                trNumThreadsVidi = ...
                    length(ar.prop.tr_f.tr_threads_with_fv{i});
                for j = 1:trNumThreadsVidi
                    tKey = sprintf('%08d_%08d', i, ...
                        ar.prop.tr_f.tr_threads_with_fv{i}{j});
                    tKeySet{end+1} = tKey;
                    rCounter = rCounter + 1;
                end
            end
            ar.prop.trThread2rowMap = containers.Map(tKeySet,tValueSet);
         end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
        function mapTeThreadNum(ar)
            % Map thread numbers for lookup

            rCounter = 1;
            tKeySet = {};
            tValueSet = 1:ar.prop.num_te;
            for i = 1:ar.prop.num_te_video
                teNumThreadsVidi = ...
                    length(ar.prop.te_f.te_threads_with_fv{i});
                for j = 1:teNumThreadsVidi
                    tKey = sprintf('%08d_%08d', i, ...
                        ar.prop.te_f.te_threads_with_fv{i}{j});
                    tKeySet{end+1} = tKey;
                    rCounter = rCounter + 1;
                end
            end
            ar.prop.teThread2rowMap = containers.Map(tKeySet,tValueSet);
         end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function createTrainSet(ar)
        ar.prop.numTrainSamplesVideo = zeros(ar.prop.num_tr_video,1);
        ar.prop.numTrainSamples = 0;
        for v = 1:ar.prop.num_tr_video
            n = length(ar.prop.tr_f.tr_threads_with_fv{v});
            ar.prop.numTrainSamplesVideo(v) = 0;
            lb = max(1,n-ar.prop.subset_size_ub+1);
            for k = n:-1:lb
                ar.prop.numTrainSamplesVideo(v) = ...
                    ar.prop.numTrainSamplesVideo(v) + nchoosek(n,k);
            end
            ar.prop.numTrainSamples = ar.prop.numTrainSamples + ...
                ar.prop.numTrainSamplesVideo(v);
        end

        ar.prop.trXavg = zeros(ar.prop.num_tr_video,ar.prop.feat_dim);
        trXCtr = 1;
        ar.prop.trX = zeros(ar.prop.numTrainSamples,ar.prop.feat_dim);
        for v = 1:ar.prop.num_tr_video

%            if v == 700
%                keyboard
%            end

            % Obtain the index of each thread in the train set
            trThreadIndexRelVidv = ar.prop.tr_f.tr_threads_with_fv{v};
            trNumThreadsVidv = length(trThreadIndexRelVidv);
            trThreadIndexVidv = zeros(trNumThreadsVidv,1);
            for t = 1:trNumThreadsVidv
                tt = trThreadIndexRelVidv{t};
                threadKey = sprintf('%08d_%08d', v, tt);
                trThreadIndexVidv(t,:) = ar.prop.trThread2rowMap(threadKey);
%                trThreadIndexHmapVSVidv(tt) = t;
            end

            %fprintf('Video %d\n', v);
            %disp(trThreadIndexVidv);
            %fprintf('________________\n');
            ar.prop.trXavg(v,:) = normavg( ...
                ar.prop.tr_f.train_fv, ...
                trThreadIndexVidv);
            %keyboard

%            trThreadIndexHmapVidv = ...
%                containers.Map(1:trNumThreadsVidv,trThreadIndexHmapVSVidv);

            % Calculate cumsum for DP
%            cumsumVidv = cumsum( ...
%                ar.prop.tr_f.train_fv(trThreadIndexVidv,:),1);
%            cumsumZVidv = [ zeros(1, ar.prop.feat_dim) ; cumsumVidv ];

            % Compute subsets of size N, N-1, ... ; N: num of threads
            %trThreadSubsetsVidv = get_subsets(trThreadIndexRelVidv, ...
%            trThreadSubsetsVidv = get_subsets(1:trNumThreadsVidv, ...
%                ar.prop.subset_size_ub);
%
            trThreadSubsetsVidv = get_subsets(trThreadIndexVidv, ...
                ar.prop.subset_size_ub);
            %disp(trThreadSubsetsVidv);

            % Normalized average features
            numSz = length(trThreadSubsetsVidv);
            for sz = 1:numSz
                lenSz = size(trThreadSubsetsVidv{sz},1);
                for i = 1:lenSz
%                    ar.prop.trX(trXCtr,:) = ... 
%                        normavg2(cumsumZVidv,trThreadSubsetsVidv{sz}(i,:)); 
                    ar.prop.trX(trXCtr,:) = ... 
                        normavg(ar.prop.tr_f.train_fv, ...
                        trThreadSubsetsVidv{sz}(i,:)); 
                    %keyboard
%                    if and(sz == 1,  i == 1)
%                        fprintf('%d  %d\n', ...
%                            length(ar.prop.tr_f.tr_threads_with_fv{v}), ...
%                            length(trThreadSubsetsVidv{sz}(i,:)))
%                        ar.prop.trXavg(v,:) = ar.prop.trX(trXCtr,:);
%                    end
                    trXCtr = trXCtr + 1;
                    if mod(trXCtr, 1000) == 0
                        fprintf('%d\n', trXCtr);
                    end
                end
            end
        end
        fprintf('Total Train Set size:\n');
        size(ar.prop.trX)
        ar.prop.trNumSubsets = size(ar.prop.trX,1);

        ar.prop.trX = [ ar.prop.trX ; ar.prop.tr_f_video.train_fv ];
        fprintf('Total Train Set size - augemented:\n');
        size(ar.prop.trX)

    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function createTestSet(ar)

        ar.prop.numTestSamplesVideo = zeros(ar.prop.num_te_video,1);
        ar.prop.numTestSamples = 0;
        for v = 1:ar.prop.num_te_video
            n = length(ar.prop.te_f.te_threads_with_fv{v});
            ar.prop.numTestSamplesVideo(v) = 0;
            lb = max(1,n-ar.prop.subset_size_ub);
            for k = n:-1:lb
                ar.prop.numTestSamplesVideo(v) = ...
                    ar.prop.numTestSamplesVideo(v) + nchoosek(n,k);
            end
            ar.prop.numTestSamples = ar.prop.numTestSamples + ...
                ar.prop.numTestSamplesVideo(v);
        end

        ar.prop.teXavg = zeros(ar.prop.num_te_video,ar.prop.feat_dim);
        teXCtr = 1;
        ar.prop.teX = zeros(ar.prop.numTestSamples,ar.prop.feat_dim);
        for v = 1:ar.prop.num_te_video

            % Obtain the index of each thread in the train set
            teThreadIndexRelVidv = ar.prop.te_f.te_threads_with_fv{v};
            teNumThreadsVidv = length(teThreadIndexRelVidv);
            teThreadIndexVidv = zeros(teNumThreadsVidv,1);
            for t = 1:teNumThreadsVidv
                tt = teThreadIndexRelVidv{t};
                threadKey = sprintf('%08d_%08d', v, tt);
                teThreadIndexVidv(t,:) = ar.prop.teThread2rowMap(threadKey);
            end


            %disp(teThreadIndexVidv);
            ar.prop.teXavg(v,:) = normavg( ...
                ar.prop.te_f.test_fv, ...
                teThreadIndexVidv);

            % Calculate cumsum for DP
%            cumsumVidv = cumsum( ...
%                ar.prop.te_f.test_fv(teThreadIndexVidv,:),1);
%            cumsumZVidv = [ zeros(1, ar.prop.feat_dim) ; cumsumVidv ];
%            teThreadSubsetsVidv = get_subsets(1:teNumThreadsVidv, ...
%                ar.prop.subset_size_ub);

            % Compute subsets of size N, N-1, ... ; N: num of threads
            teThreadSubsetsVidv = get_subsets(teThreadIndexVidv, ...
                ar.prop.subset_size_ub);

            % Normalized average features
            numSz = length(teThreadSubsetsVidv);
            for sz = 1:numSz
                lenSz = size(teThreadSubsetsVidv{sz},1);
                for i = 1:lenSz
%
%                    ar.prop.teX(teXCtr,:) = ... 
%                        normavg2(cumsumZVidv,teThreadSubsetsVidv{sz}(i,:)); 
%
                    ar.prop.teX(teXCtr,:) = ... 
                        normavg(ar.prop.te_f.test_fv, ...
                        teThreadSubsetsVidv{sz}(i,:)); 

%                    if and(sz == 1,  i == 1)
%                        ar.prop.teXavg(v,:) = ar.prop.teX(teXCtr,:);
%                    end
                    teXCtr = teXCtr + 1;
                    if mod(teXCtr, 1000) == 0
                        fprintf('%d\n', teXCtr);
                    end
                end
            end
        end
        fprintf('Total Test size:\n');
        size(ar.prop.teX)
        ar.prop.teNumSubsets = size(ar.prop.teX,1);

        ar.prop.teX = [ ar.prop.teX ; ar.prop.te_f_video.test_fv ];
        fprintf('Total Test size - augmented:\n');
        size(ar.prop.teX)
        %keyboard

    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function createCodebook(ar)
    
        t_s = tic;
        ar.prop.gmm_level2_dir = [ar.prop.base_dir '/' 'gmm_level2_dir' ];
        mkdir(ar.prop.gmm_level2_dir);
        dataset = 'Improved_Traj';
        X = [ ar.prop.tr_f.train_fv ; ar.prop.tr_f_video.train_fv ];
        save_file = [ ar.prop.gmm_level2_dir '/' 'gmm_level2.mat' ];
        [means, covariances, priors] = ...
            run_gmm(X, dataset, save_file);
        t_e = toc(t_s);
        fprintf('Create Codebook: %f sec\n', t_e);
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function createLevel2TrainFV(ar)
        
        t_s = tic;
        ar.prop.gmm_level2_dir = [ar.prop.base_dir '/' 'gmm_level2_dir' ];
        load_file = [ ar.prop.gmm_level2_dir '/' 'gmm_level2.mat' ];
        gmm2 = load(load_file);
        
        ar.prop.fv_level2_dir = [ar.prop.base_dir '/' 'fisher_level2_dir' ];
        mkdir(ar.prop.fv_level2_dir);
        save_file = [ar.prop.fv_level2_dir '/' 'fisher_level2_train.mat'];

        ar.prop.fvL2Len = 2 * size(gmm2.means,1) * size(gmm2.means,2);
        trXL2 = zeros(ar.prop.num_tr_video, ar.prop.fvL2Len);
        for v = 1:ar.prop.num_tr_video
            trThreadIndexRelVidv = ar.prop.tr_f.tr_threads_with_fv{v};
            trNumThreadsVidv = length(trThreadIndexRelVidv);
            trThreadIndexVidv = zeros(trNumThreadsVidv,1);
            trXL2v = zeros(trNumThreadsVidv+1,ar.prop.feat_dim);
            for t = 1:trNumThreadsVidv
                tt = trThreadIndexRelVidv{t};
                threadKey = sprintf('%08d_%08d', v, tt);
                trThreadIndexVidv(t,:) = ar.prop.trThread2rowMap(threadKey);
                trXL2v(t,:) = ar.prop.tr_f.train_fv(trThreadIndexVidv(t,:),:);
            end
            trXL2v(end,:) = ar.prop.tr_f_video.train_fv(v,:);
            trXL2(v,:) = vl_fisher(trXL2v', ...
                gmm2.means, ...
                gmm2.covariances, ...
                gmm2.priors, ...
                'Improved');  
        end

        disp(size(trXL2));
        save(save_file, '-v7.3', 'trXL2');
        t_e = toc(t_s);
        fprintf('Feature vector creation (train): %f sec\n', t_e);
    end


    function createLevel2TestFV(ar)
        
        t_s = tic;
        ar.prop.gmm_level2_dir = [ar.prop.base_dir '/' 'gmm_level2_dir' ];
        load_file = [ ar.prop.gmm_level2_dir '/' 'gmm_level2.mat' ];
        gmm2 = load(load_file);
        
        ar.prop.fv_level2_dir = [ar.prop.base_dir '/' 'fisher_level2_dir' ];
        mkdir(ar.prop.fv_level2_dir);
        save_file = [ar.prop.fv_level2_dir '/' 'fisher_level2_test.mat'];

        ar.prop.fvL2Len = 2 * size(gmm2.means,1) * size(gmm2.means,2);
        teXL2 = zeros(ar.prop.num_te_video, ar.prop.fvL2Len);
        for v = 1:ar.prop.num_te_video
            teThreadIndexRelVidv = ar.prop.te_f.te_threads_with_fv{v};
            teNumThreadsVidv = length(teThreadIndexRelVidv);
            teThreadIndexVidv = zeros(teNumThreadsVidv,1);
            teXL2v = zeros(teNumThreadsVidv+1,ar.prop.feat_dim);
            for t = 1:teNumThreadsVidv
                tt = teThreadIndexRelVidv{t};
                threadKey = sprintf('%08d_%08d', v, tt);
                teThreadIndexVidv(t,:) = ar.prop.teThread2rowMap(threadKey);
                teXL2v(t,:) = ar.prop.te_f.test_fv(teThreadIndexVidv(t,:),:);
            end
            teXL2v(end,:) = ar.prop.te_f_video.test_fv(v,:);
            teXL2(v,:) = vl_fisher(teXL2v', ...
                gmm2.means, ...
                gmm2.covariances, ...
                gmm2.priors, ...
                'Improved');  
        end

        disp(size(teXL2));
        save(save_file, '-v7.3', 'teXL2');
        t_e = toc(t_s);
        fprintf('Feature vector creation (test): %f sec\n', t_e);
    end


    end
end
