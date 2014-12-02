classdef VideoActionRecognizer < handle

    properties
        prop
    end

    methods 

        function ar = VideoActionRecognizer()
            
        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function recognize(ar)
            t_start = tic;  
            ar.init();
            ar.kernelComp();
            ar.classifyAllCategories();
            ar.presentResults();
            t_elapsed = toc(t_start);
            fprintf('\nTotal time for classification : %f sec\n', t_elapsed);
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function init(ar)

            ar.prop.num_par_threads = 6;
            ar.prop.run_desc = 'Convert to LSSVM trials';
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

            ar.prop.fileorder_f = [ ar.prop.src_dir '/' 'fileorder.mat' ];
            ar.prop.fileorder = load(ar.prop.fileorder_f);
            retain_frac_threads_str = ... 
                strrep(sprintf('%6.4f', ar.prop.retain_frac_threads), ...
                '.', '_');
            ar.prop.results_file = [ ar.prop.results_dir '/' rnd_out_prefix ...
                '__' 'classification__' ...
                retain_frac_threads_str '.mat'];

            fprintf('RESULTS_FILE: %s\n', ar.prop.results_file);
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        function classifyAllCategories(ar)

            for i = 1:ar.prop.num_classes
                fprintf('%s\n', ar.prop.classes{i});

                % Input params
                param(i).dset_dir            = ar.prop.dset_dir;
                param(i).base_dir            = ar.prop.base_dir;
                param(i).cl                  = ar.prop.classes{i};
                param(i).tr                  = ar.prop.tr_f;
                param(i).te                  = ar.prop.te_f;
                param(i).tr_v                = ar.prop.tr_f_video;
                param(i).te_v                = ar.prop.te_f_video;
                param(i).fileorder           = ar.prop.fileorder;
                param(i).retain_frac_threads = ar.prop.retain_frac_threads;
                param(i).Linear_K_video      = ar.prop.Linear_K_video;
                param(i).Linear_KK_video     = ar.prop.Linear_KK_video;
                param(i).Linear_K            = ar.prop.Linear_K;
                param(i).Linear_KK           = ar.prop.Linear_KK;
                param(i).subset_size_ub      = 2;

                % Call our function
                classifier(i) = Classifier(param(i));
                ar.prop.res(i) = classifier(i).classifyByThreadSelection();

            end
        end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function presentResults(ar)
            ar.prop.mAP.lssvm_video_baseline = ...
                calc_mean_ap(ar.prop.classes, ...
                { ar.prop.res(:).lssvm_video_baseline });
            ar.prop.mAP.lssvm_thread_baseline = ... 
                calc_mean_ap(ar.prop.classes, ...
                { ar.prop.res(:).lssvm_thread_baseline });

            fprintf('\n');
            fprintf('Results : Mean Average Precision\n');
            fprintf('%-25s : %15f\n', 'Video baseline', ...
                ar.prop.mAP.lssvm_video_baseline);
            fprintf('%-25s : %15f\n', 'Thread baseline', ...
                ar.prop.mAP.lssvm_thread_baseline);
            fprintf('\n');


%            save(ar.prop.results_file, ar.prop.base_dir, ...
%                ar.prop.retain_frac_threads, ...
%                ar.prop.num_classes, ar.prop.classes, ...
%                ar.prop.res, ar.prop.mAP);

        end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
end
