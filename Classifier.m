classdef Classifier < handle
    properties
        param
        out
    end
    methods

        function classifier = Classifier(param)
            if nargin > 0
                classifier.param = param;
            end
            run('/nfs/bigeye/sdaptardar/installs/vlfeat/toolbox/vl_setup')
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function out = classifyByThreadSelection(classifier)
            classifier.parseLabels(); 
            classifier.initVars();
            classifier.videoLevelClassifier();
            %classifier.threadLevelClassifier();
            %classifier.augLevelClassifier();
            classifier.normAvgLevelClassifier();
            classifier.bestSubsetClassifier();
            out = classifier.out;
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function parseLabels(classifier)
            % Parse labels file for training set

            % video
            labels_dict_file_train = ...
                sprintf('%s%s%s%s%s%s%s%s', ...
                classifier.param.dset_dir, '/', 'ClipSets', '/', ...
                classifier.param.cl, '_', 'train', '.txt');
            fprintf('%s\n', labels_dict_file_train); 
            [classifier.out.training_labels_fname_video, ...
                classifier.out.training_labels_vector_video] = ...
                textread(labels_dict_file_train, '%s %d');
            tr_sz = size(classifier.param.tr.train_fv);
            classifier.out.num_tr = tr_sz(1);

            % threads
            tr_fname = cell(classifier.out.num_tr,1);
            classifier.out.training_labels_vector = ...
                zeros(classifier.out.num_tr,1);
            for i = 1:classifier.out.num_tr
                tr_fname{i} = ...
                char(classifier.out.training_labels_fname_video( ...
                    classifier.param.tr.tr_r2t(i).fileNum,:));
                classifier.out.training_labels_vector(i,1) = ...
                    classifier.out.training_labels_vector_video( ...
                    classifier.param.tr.tr_r2t(i).fileNum, :);
            end
            classifier.out.training_labels_fname = char(tr_fname);

            % thread subsets + videos augmented training set
            classifier.out.num_tr_aug = size(classifier.param.trX, 1);
            classifier.out.training_labels_vector_aug = ...
                zeros(classifier.out.num_tr_aug,1);
            classifier.out.training_labels_fname_aug  = ...
                cell(classifier.out.num_tr_aug,1);
            lblIdx = 1;
            classifier.out.num_tr_video = size( ...
                classifier.out.training_labels_vector_video,1);
            for i = 1:classifier.out.num_tr_video
%                classifier.out.training_labels_vector_aug( ...
%                lblIdx:lblIdx+classifier.param.numTrainSamplesVideo(i),:)...
%                 = repmat([classifier.out.training_labels_vector_video(i)],...
%                             classifier.param.numTrainSamplesVideo(i),1);

                for j = 1:classifier.param.numTrainSamplesVideo(i)
                    classifier.out.training_labels_vector_aug(lblIdx+j-1) = ...
                        classifier.out.training_labels_vector_video(i);
                end

                for j = 1:classifier.param.numTrainSamplesVideo(i)
                    classifier.out.training_labels_fname_aug(lblIdx+j-1) = ...
                        classifier.out.training_labels_fname_video(i,:);
                end

                lblIdx = lblIdx + classifier.param.numTrainSamplesVideo(i);
            end

            fprintf('lblIdx  %d\n', lblIdx);
            size(classifier.out.training_labels_vector_aug)
            size(classifier.out.training_labels_vector_video)
%            classifier.out.training_labels_vector_aug(lblIdx:end,:) = ...
%                   classifier.out.training_labels_vector_video(:);
            
            % Parse labels file for testing set
            labels_dict_file_test = ...
                sprintf('%s%s%s%s%s%s%s%s', ...
                classifier.param.dset_dir, '/', 'ClipSets', '/', ...
                classifier.param.cl, '_', 'test', '.txt');
            fprintf('%s\n', labels_dict_file_test); 
            [classifier.out.testing_labels_fname_video, ...
                classifier.out.testing_labels_vector_video] = ...
                textread(labels_dict_file_test, '%s %d');
            te_sz = size(classifier.param.te.test_fv);
            classifier.out.num_te = te_sz(1);
            te_fname = cell(classifier.out.num_te,1);
            classifier.out.testing_labels_vector = ...
                zeros(classifier.out.num_te,1);
            for i = 1:classifier.out.num_te
                te_fname{i} = ...
                    char(classifier.out.testing_labels_fname_video( ...
                    classifier.param.te.te_r2t(i).fileNum,:));
                classifier.out.testing_labels_vector(i,1) = ...
                    classifier.out.testing_labels_vector_video( ...
                    classifier.param.te.te_r2t(i).fileNum, :);
            end
            classifier.out.testing_labels_fname = char(te_fname);


            % thread subsets + videos augmented test set
            classifier.out.num_te_video = size( ...
                classifier.out.testing_labels_vector_video,1);
            classifier.out.num_te_aug = size(classifier.param.teX, 1);
            classifier.out.testing_labels_vector_aug = ...
                zeros(classifier.out.num_te_aug,1);
            classifier.out.testing_labels_fname_aug  = ...
                cell(classifier.out.num_te_aug,1);
            lblIdx = 1;
            for i = 1:classifier.out.num_te_video
%                classifier.out.testing_labels_vector_aug( ...
%                lblIdx:lblIdx+classifier.param.numTestSamplesVideo(i)-1,:)...
%                    = repmat([classifier.out.testing_labels_vector_video(i)],...
%                             classifier.param.numTestSamplesVideo(i),1);

                for j = 1:classifier.param.numTestSamplesVideo(i)
                    classifier.out.testing_labels_vector_aug(lblIdx+j-1) = ...
                        classifier.out.testing_labels_vector_video(i);
                end

                for j = 1:classifier.param.numTestSamplesVideo(i)
                    classifier.out.testing_labels_fname_aug(lblIdx+j-1) = ...
                        classifier.out.testing_labels_fname_video(i,:);
                end

                lblIdx = lblIdx + classifier.param.numTestSamplesVideo(i);
            end

%            classifier.out.testing_labels_vector_aug(lblIdx:end,:) = ...
%                   classifier.out.testing_labels_vector_video(:);

        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
        function videoLevelClassifier(classifier)
            % Train a video level classifier

            t_cv_start_video = tic;

            lssvm.lambda = 10^-3 * classifier.out.num_tr_video; 

            [lssvm.alphas,lssvm.b, lssvm.cvErrs, lssvm.cvAlphas, lssvm.cvBs] = ...
                ML_Ridge.kerRidgeReg_cv(classifier.param.Linear_K_video, ...
                classifier.out.training_labels_vector_video, lssvm.lambda, ...
                ones(classifier.out.num_tr_video,1));

            lssvm.w = classifier.param.Linear_K_video * lssvm.alphas;            
            lssvm.decision_values = ...
                classifier.param.Linear_KK_video * lssvm.alphas + lssvm.b; 

           [lssvm.recall, lssvm.precision, lssvm.ap_info] = ...
               vl_pr(classifier.out.testing_labels_vector_video, ...
               lssvm.decision_values);

            lssvm.rmse = ...
                sqrt( (norm(lssvm.cvErrs)^2) / classifier.out.num_tr_video );

            classifier.out.lssvm_video_baseline = lssvm;

            t_cv_elapsed_video = toc(t_cv_start_video);

            fprintf('LSSVM (video): %s : %10f\t time = %10f \n', ...
                classifier.param.cl, ...
                lssvm.rmse, t_cv_elapsed_video);
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
        function threadLevelClassifier(classifier)
            % Train a thread level classifier (use labels of videos)

            t_ct_start = tic;

            lssvm.lambda = 10^-3 * classifier.out.num_tr; 

            [lssvm.alphas, lssvm.b, lssvm.cvErrs, lssvm.cvAlphas, lssvm.cvBs] = ...
                ML_Ridge.kerRidgeReg_cv(classifier.param.Linear_K, ...
                classifier.out.training_labels_vector, lssvm.lambda, ...
                ones(classifier.out.num_tr,1));

            lssvm.w = classifier.param.Linear_K * lssvm.alphas;            
            lssvm.decision_values = ...
                classifier.param.Linear_KK * lssvm.alphas + lssvm.b; 

           [lssvm.recall, lssvm.precision, lssvm.ap_info] = ...
               vl_pr(classifier.out.testing_labels_vector, ...
               lssvm.decision_values);

            lssvm.rmse = sqrt( (norm(lssvm.cvErrs)^2) / classifier.out.num_tr);

            classifier.out.lssvm_thread_baseline = lssvm;

            t_ct_elapsed = toc(t_ct_start);

            fprintf('LSSVM (threads): %s :  %10f\t time = %10f \n', ...
                classifier.param.cl, ...
                lssvm.rmse, t_ct_elapsed);
        end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function augLevelClassifier(classifier)
            % Train a video level classifier

            t_cv_start_aug = tic;

            lssvm.lambda = 10^-3 * classifier.out.num_tr_aug; 

            [lssvm.alphas,lssvm.b, lssvm.cvErrs, lssvm.cvWs, lssvm.cvBs] = ...
                ML_Ridge.kerRidgeReg_cv(classifier.param.Linear_K_aug, ...
                classifier.out.training_labels_vector_aug, lssvm.lambda, ...
                ones(classifier.out.num_tr_aug,1));

            lssvm.w = classifier.param.Linear_K_aug * lssvm.alphas;            
            lssvm.decision_values = ...
                classifier.param.Linear_KK_aug * lssvm.alphas + lssvm.b; 

           [lssvm.recall, lssvm.precision, lssvm.ap_info] = ...
               vl_pr(classifier.out.testing_labels_vector_aug, ...
               lssvm.decision_values);

            lssvm.rmse = ...
                sqrt( (norm(lssvm.cvErrs)^2) / classifier.out.num_tr_aug );

            classifier.out.lssvm_aug_baseline = lssvm;

            t_cv_elapsed_aug = toc(t_cv_start_aug);

            fprintf('LSSVM (Aug): %s : %10f\t time = %10f \n', ...
                classifier.param.cl, ...
                lssvm.rmse, t_cv_elapsed_aug);
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function normAvgLevelClassifier(classifier)
            
            t_at_start = tic;

            % Parmaeters
            lssvm.lambda = 10^-3 * classifier.out.num_tr_video; 

            % Build Model
            [lssvm.alphas, lssvm.b, lssvm.cvErrs, lssvm.cvAlphas, lssvm.cvBs] = ...
                ML_Ridge.kerRidgeReg_cv(classifier.param.Linear_K_avg, ...
                classifier.out.training_labels_vector_video, lssvm.lambda, ...
                ones(classifier.out.num_tr_video,1));

            lssvm.w = classifier.param.Linear_K_avg * lssvm.alphas;            

            % Decision values and error estimates 
            lssvm.decision_values = ...
                classifier.param.Linear_KK_avg * lssvm.alphas + lssvm.b; 

           [lssvm.recall, lssvm.precision, lssvm.ap_info] = ...
               vl_pr(classifier.out.testing_labels_vector_video, ...
               lssvm.decision_values);

            lssvm.rmse = ...
                sqrt( (norm(lssvm.cvErrs)^2) / classifier.out.num_tr_video);

            classifier.out.lssvm_normavg_baseline = lssvm;

            t_at_elapsed = toc(t_at_start);

            fprintf('LSSVM (Norm Avg): %s :  %10f\t time = %10f \n', ...
                classifier.param.cl, ...
                lssvm.rmse, t_at_elapsed);

        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function bestSubsetClassifier(classifier)

%            classifier.param.Linear_K_avg = ...
%                classifier.param.trXavg * classifier.param.trXavg';
%            classifier.param.Linear_KK_avg = ...
%                classifier.param.teXavg * classifier.param.trXavg';
%
%
%            classifier.param.Linear_K_val = ...
%                classifier.param.trX * classifier.param.trXavg';


            lssvm.lambda = 10^-3 * classifier.out.num_tr_video; 

            % Apply LSSVM
%           [lssvm.alphas,lssvm.b, lssvm.cvErrs, lssvm.cvAlphas, lssvm.cvBs] = ...
%                ML_Ridge.kerRidgeReg_cv(classifier.param.Linear_K_video, ...
%                classifier.out.training_labels_vector_video, lssvm.lambda, ...
%                ones(classifier.out.num_tr_video,1));
%
%%            [lssvm.alphas,lssvm.b,lssvm.cvErrs,lssvm.cvAlphas,lssvm.cvBs] = ...
%%            ML_Ridge.kerRidgeReg_cv(classifier.param.Linear_K_avg, ...
%%                classifier.out.training_labels_vector_video, lssvm.lambda, ...
%%                ones(classifier.out.num_tr_video,1));
%
%            %lssvm.cvWs = classifier.param.Linear_K_val * lssvm.cvAlphas;            
%            % Select the thread subset or the video with highest 
%            % cross-validation score 
%
%            classifier.out.trXS = zeros(classifier.out.num_tr_video, ...
%                                        classifier.out.feat_dim);
%            trIdx = 1;
%            for i = 1:classifier.out.num_tr_video
%                %k = classifier.param.numTrainSamples + i;
%%                k = 0;
%                mx = -Inf;
%                mxIdx = 0;
%                if classifier.out.training_labels_vector_video(i) == 1
%                    for j = 1:classifier.param.numTrainSamplesVideo(i)
%                        t = trIdx+j-1;
%                        dvS = ...
%                        classifier.param.Linear_K_val(t,:)*lssvm.cvAlphas(:,i) ...
%                            + lssvm.cvBs(i);
%                        dvS = classifier.out.training_labels_vector_video(i) ...
%                            * dvS;
%                        if dvS > mx
%                            mx = dvS;
%                            mxIdx = t;
%                        end
%                    end
%                    t = classifier.param.numTrainSamples+i;
%                    dvV = classifier.param.Linear_K_val(t,:)*lssvm.cvAlphas(:,i)...
%                        + lssvm.cvBs(i);
%                    dvV = classifier.out.training_labels_vector_video(i) ...
%                            * dvV;
%                    if dvV > mx
%                        mx = dvV;
%                        mxIdx = t;
%                    end
%                    k = mxIdx;
%                    classifier.out.trXS(i,:) = classifier.param.trX(mxIdx,:);
%                else
%                    classifier.out.trXS(i,:) = ...
%                        classifier.param.tr_v.train_fv(i,:);
%                    %    classifier.param.trXavg(i,:);
%                end
%                trIdx = trIdx + classifier.param.numTrainSamplesVideo(i);
%            end
%            
%%           [lssvm.recall, lssvm.precision, lssvm.ap_info] = ...
%%               vl_pr(classifier.out.testing_labels_vector_video, ...
%%               lssvm.decision_values);
%%
%%            lssvm.rmse = ...
%%                sqrt( (norm(lssvm.cvErrs)^2) / classifier.out.num_tr_video );
%%
%%            classifier.out.lssvm_video_baseline = lssvm;
%
%%            t_cv_elapsed_video = toc(t_cv_start_video);
%
%%            fprintf('LSSVM (video): %s : %10f\t time = %10f \n', ...
%%                classifier.param.cl, ...
%%                lssvm.rmse, t_cv_elapsed_video);


            classifier.out.trXS = classifier.param.trXavg;
            classifier.out.teXS = classifier.param.teXavg;
            %classifier.out.trXS = classifier.param.tr_v.train_fv;
%            classifier.out.teXS = classifier.param.te_v.test_fv;

            % Calculate the Kernel matrices 
            classifier.out.Linear_K_XS = ...
                classifier.out.trXS * classifier.out.trXS';

%            classifier.out.Linear_KK_XS = ...
%                classifier.param.te_v.test_fv * classifier.out.trXS';

            classifier.out.Linear_KK_XS = ...
                classifier.out.teXS * classifier.out.trXS';

             %keyboard
%            classifier.out.Linear_KK_XS = ...
%                classifier.param.teXavg * classifier.out.trXS';

            % Build model
            [lssvmXS.alphas, lssvmXS.b, lssvmXS.cvErrs, ...
                lssvmXS.cvAlphas, lssvmXS.cvBs] = ...
                ML_Ridge.kerRidgeReg_cv(classifier.out.Linear_K_XS, ...
                classifier.out.training_labels_vector_video, lssvm.lambda, ...
                ones(classifier.out.num_tr_video,1));

            % Classify

            lssvmXS.decision_values = ...
                classifier.out.Linear_KK_XS * lssvmXS.alphas + lssvmXS.b; 

           [lssvmXS.recall, lssvmXS.precision, lssvmXS.ap_info] = ...
               vl_pr(classifier.out.testing_labels_vector_video, ...
               lssvmXS.decision_values);

            classifier.out.lssvmXS = lssvmXS;

            %keyboard
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
    function initVars(classifier) 
        %  Some declarations

            classifier.out.num_tr_pos_video = ... 
                sum( classifier.out.training_labels_vector_video == 1);
            classifier.out.num_tr_pos = ...
                sum(classifier.out.training_labels_vector == 1);
            classifier.out.num_tr_neg_video = ...
                sum(classifier.out.training_labels_vector_video == -1);
            classifier.out.num_tr_neg = ...
                sum(classifier.out.training_labels_vector == -1);
            %classifier.out.loocvscore = zeros(num_tr_pos,1);
            classifier.out.feat_dim = size(classifier.param.tr.train_fv,2);
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
end
