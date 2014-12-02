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
            classifier.mapThreadNum();
            classifier.videoLevelClassifier();
            classifier.threadLevelClassifier();
            classifier.classifyByBestThreadSubsetClassifier();
            out = classifier.out;
        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        function parseLabels(classifier)
            % Parse labels file for training set

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
            tr_fname = {};
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
            te_fname = {};
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

        end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
        function videoLevelClassifier(classifier)
            % Train a video level classifier

            t_cv_start_video = tic;

            lssvm.lambda = 10^-3 * classifier.out.num_tr_video; 

            [lssvm.alphas,lssvm.b, lssvm.cvErrs, lssvm.cvWs, lssvm.cvBs] = ...
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

            [lssvm.alphas, lssvm.b, lssvm.cvErrs, lssvm.cvWs, lssvm.cvBs] = ...
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

        function mapThreadNum(classifier)
            % Map thread numbers for lookup

            rCounter = 1;
            tKeySet = {};
            tValueSet = 1:classifier.out.num_tr;
            for i = 1:classifier.out.num_tr_video
                trNumThreadsVidi = ...
                    length(classifier.param.tr.tr_threads_with_fv{i});
                for j = 1:trNumThreadsVidi
                    tKey = sprintf('%8d_%8d', i, ...
                        classifier.param.tr.tr_threads_with_fv{i}{j});
                    tKeySet{end+1} = tKey;
                    rCounter = rCounter + 1;
                end
            end
            classifier.out.thread2rowMap = containers.Map(tKeySet,tValueSet);
         end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    function initVars(classifier) 
        %  Some declarations

            classifier.out.num_tr_video = size( ...
                classifier.out.training_labels_vector_video,1);
            classifier.out.num_te_video = size( ...
                classifier.out.testing_labels_vector_video,1);
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


    function classifyByBestThreadSubsetClassifier(classifier)

        classifier.out.numTrain = 0;
        for v = 1:classifier.out.num_tr_video
            n = length(classifier.param.tr.tr_threads_with_fv{v});
            for k = n:n-classifier.param.subset_size_ub
                classifier.out.numTrain = ...
                    classifier.out.numTrain + nchoosek(n,k);
            end
        end

        trXCtr = 1;
        trX = zeros(classifier.out.numTrain,classifier.out.feat_dim);
        for v = 1:classifier.out.num_tr_video

            % Obtain the index of each thread in the train set
            trThreadIndexRelVidv = classifier.param.tr.tr_threads_with_fv{v};
            trNumThreadsVidv = length(trThreadIndexRelVidv);
            trThreadIndexVidv = zeros(trNumThreadsVidv,1);
            for t = 1:trNumThreadsVidv
                tt = trThreadIndexRelVidv{t};
                threadKey = sprintf('%8d_%8d', v, tt);
                trThreadIndexVidv(t,:)=classifier.out.thread2rowMap(threadKey);
%                trThreadIndexHmapVSVidv(tt) = t;
            end
%            trThreadIndexHmapVidv = ...
%                containers.Map(1:trNumThreadsVidv,trThreadIndexHmapVSVidv);

            % Calculate cumsum for DP
            cumsumVidv = cumsum( ...
                classifier.param.tr.train_fv(trThreadIndexVidv,:),1);
            cumsumZVidv = [ zeros(1, classifier.out.feat_dim) ; cumsumVidv ];

            % Compute subsets of size N, N-1, ... ; N: num of threads
            %trThreadSubsetsVidv = get_subsets(trThreadIndexRelVidv, ...
            trThreadSubsetsVidv = get_subsets(1:trNumThreadsVidv, ...
                classifier.param.subset_size_ub);
            %disp(trThreadSubsetsVidv);

            % Normalized average features
            numSz = length(trThreadSubsetsVidv);
            for sz = 1:numSz
                lenSz = size(trThreadSubsetsVidv{sz},1);
                for i = 1:lenSz
                    %    trX = [trX ; normavg(classifier.param.tr.train_fv, ...
                    %        trThreadSubsetsVidv{sz}(i))];

%                    disp(trThreadSubsetsVidv{sz}(i,:));
%                    trX = [ trX ; ...
%                        normavg2(cumsumZVidv,trThreadSubsetsVidv{sz}(i,:))]; 
%                    size(trX)
%
                    trX(trXCtr,:) = ... 
                        normavg2(cumsumZVidv,trThreadSubsetsVidv{sz}(i,:)); 
                    trXCtr = trXCtr + 1;
                    if mod(trXCtr, 1000) == 0
                        fprintf('%d\n', trXCtr);
                    end
                end
            end
        end
        fprintf('Total size\n');
        size(trX)
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% % Find LOOCV thread for every thread
        %% 
        %%     thread_based_features_tr_video = zeros(0,feat_dim);
        %%     Linear_KK = classifier.param.tr.train_fv * classifier.param.tr_v.train_fv';   % kernel for validation
        %%     weight_neg = num_tr_pos_video-1;
        %%     weight_pos = num_tr_neg_video-1;
        %%     opt = sprintf('-c 100 -w1 %f -w-1 %f -t 4', weight_pos, weight_neg);
        %%     tr_pos_counter = 1;
        %%     tr_ix_pruned = zeros(0,1);
        %%     num_pos_reduced = 0;
        %%     for v = 1:classifier.out.num_tr_video                                  % for every thread
        %%         num_tr_vid_v = length(classifier.param.tr.tr_threads_with_fv{v});
        %%         if classifier.out.training_labels_vector_video(v,:) == 1                                                     % if +ve
        %%             K = classifier.param.Linear_K_video;                                                                       % drop the video fv
        %%             K(v,:) = [];
        %%             K(:,v) = [];
        %%             L = classifier.out.training_labels_vector_video;                                                         % and labels
        %%             L(v,:) = [];
        %%             cv_auc(tr_pos_counter,1) = do_binary_cross_validation(L, [ (1:classifier.out.num_tr_video-1)' K ], opt, n_fold);
        %%             fprintf('%15s : cv_auc : %5d : %6f\n', classifier.param.cl, v, cv_auc(tr_pos_counter,1));
        %%             model_leaveoneout = svmtrain(L, [ (1:classifier.out.num_tr_video-1)' K], opt); 
        %%             val_ix = zeros(num_tr_vid_v,1);
        %%             for j = 1:num_tr_vid_v
        %%                 t = classifier.param.tr.tr_threads_with_fv{v}{j};
        %%                 thread_key = sprintf('%8d_%8d', v, t);
        %%                 val_ix(j,:) = thread2row_map(thread_key);
        %%             end
        %%             KK = Linear_KK(val_ix,:);
        %%             KK(:,v) = [];
        %%             [pr_lbl, acc, dec_val] = svmpredict(ones(num_tr_vid_v,1), [ (1:num_tr_vid_v)' KK], model_leaveoneout); 
        %%             disp(dec_val);
        %%             [dec_val_sorted, dec_val_sorted_ix] = sort(dec_val, 'descend'); % sort +ve threads by loocv score ranks 
        %%             %n = max(round(classifier.param.retain_frac_threads * num_tr_vid_v), 1); 
        %%             n = max( num_tr_vid_v-1, 1);   % all but one 
        %%             num_pos_reduced = num_pos_reduced + n;
        %%             sorted_val_ix = val_ix(dec_val_sorted_ix,:);
        %%             sorted_val_ix_reduced = sorted_val_ix(1:n,:);
        %%             avg_feature_video = mean(classifier.param.tr.train_fv(sorted_val_ix_reduced,:), 1);
        %%             nm_avg_feature_video = norm(avg_feature_video);
        %%             if nm_avg_feature_video ~= 0
        %%                 avg_feature_video = avg_feature_video / nm_avg_feature_video; 
        %%             else
        %%                 avg_feature_video = zeros(1, feat_dim);
        %%             end
        %%             thread_based_features_tr_video = [ thread_based_features_tr_video ; avg_feature_video ];
        %%             %tr_ix_pruned = [ tr_ix_pruned ; sorted_val_ix(1:n,:) ]; 
        %%             tr_ix_pruned = [ tr_ix_pruned ; sorted_val_ix_reduced(1:n,:) ]; 
        %%             tr_pos_counter = tr_pos_counter + 1;
        %%         else
        %%             for j = 1:num_tr_vid_v
        %%                 t = classifier.param.tr.tr_threads_with_fv{v}{j};
        %%                 thread_key = sprintf('%8d_%8d', v, t);
        %%                 val_ix(j,:) = thread2row_map(thread_key);
        %%             end
        %%             avg_feature_video = mean(classifier.param.tr.train_fv(val_ix,:), 1);
        %%             nm_avg_feature_video = norm(avg_feature_video);
        %%             if nm_avg_feature_video ~= 0
        %%                 avg_feature_video = avg_feature_video / nm_avg_feature_video; 
        %%             else
        %%                 avg_feature_video = zeros(1, feat_dim);
        %%             end
        %%             thread_based_features_tr_video = [ thread_based_features_tr_video ; avg_feature_video ];
        %%         end
        %% 
        %%     end
        %%     tr_ix_pruned = [ tr_ix_pruned ; find(classifier.out.training_labels_vector == -1) ];
        %%     num_neg_reduced = num_tr_neg;
        %%     num_tr_pruned = size(tr_ix_pruned,1);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%     % Build model with irrelevant threads pruned
        %% 
        %%     %opt2 = sprintf('-c 100 -w1 %f -w-1 %f -t 4', num_neg_reduced, num_pos_reduced);  % inverse ratio of num of +ve and -ve samples
        %%     opt2 = sprintf('-c 100 -w1 %f -w-1 %f -t 4', num_tr_neg_video, num_tr_pos_video); % 1 fv per video now % inverse ratio of num of +ve and -ve samples
        %%     % Linear_K_pruned  = classifier.param.tr.train_fv(tr_ix_pruned,:) * classifier.param.tr.train_fv(tr_ix_pruned,:)';
        %%     % Linear_KK_pruned = classifier.param.te.test_fv * classifier.param.tr.train_fv(tr_ix_pruned,:)';
        %% 
        %%     Linear_K_pruned  = thread_based_features_tr_video * thread_based_features_tr_video';
        %% 
        %%     size(Linear_K_pruned)
        %%     disp(num_tr_pruned)
        %%     size((1:num_tr_pruned)')
        %%     classifier.out.model = svmtrain(classifier.out.training_labels_vector_video, [ (1:classifier.out.num_tr_video)' Linear_K_pruned], opt2); 
        %% 
        %%     num_test_video = size(classifier.param.te.te_name,1);
        %%     thread_based_features_video = zeros(num_test_video,feat_dim);
        %%     for k = 1:classifier.out.num_te_video
        %%         tfv = zeros(1, feat_dim);
        %%         for l = 1:length(classifier.param.te.te_t2r_hmap{k})
        %%             tfv = tfv + classifier.param.te.test_fv(classifier.param.te.te_t2r_hmap{k}{l}(1), :);
        %%         end
        %%         thread_based_features_video(k,:) = tfv / norm(tfv);
        %%     end
        %%     Linear_KK_tbf_video = thread_based_features_video * thread_based_features_tr_video';
        %%     disp(size(Linear_KK_tbf_video));
        %%     [classifier.out.predicted_label, classifier.out.accuracy, classifier.out.decision_values] = svmpredict(classifier.out.testing_labels_vector_video, [ (1:classifier.out.num_te_video)' Linear_KK_tbf_video], classifier.out.model); 
        %%     %[classifier.out.predicted_label, classifier.out.accuracy, classifier.out.decision_values] = svmpredict(classifier.out.testing_labels_vector, [ (1:classifier.out.num_te)' Linear_KK_pruned], classifier.out.model); 
        %%     %[predicted_label_video, accuracy_video, decision_values_video] = svmpredict(classifier.out.testing_labels_vector_video, [ (1:classifier.out.num_te_video)' Linear_KK_pruned_video], classifier.out.model); 
        %% 
        %% 
        %% %    num_test_video = size(classifier.param.te.te_name,1);
        %% %    decision_values_max_video = zeros(num_test_video,1);
        %% %    decision_values_min_video = zeros(num_test_video,1);
        %% %    decision_values_mean_video = zeros(num_test_video,1);
        %% %    decision_values_median_video = zeros(num_test_video,1);
        %% %
        %% %    for k = 1:num_test_video
        %% %        dv_k = zeros(length(classifier.param.te.te_t2r_hmap{k}),1);
        %% %        for l = 1:length(classifier.param.te.te_t2r_hmap{k})
        %% %            dv_k(l,:) = classifier.out.decision_values(classifier.param.te.te_t2r_hmap{k}{l}(1), :);
        %% %        end
        %% %        decision_values_max_video(k,:) = max(dv_k);
        %% %        decision_values_min_video(k,:) = min(dv_k);
        %% %        decision_values_median_video(k,:) = median(dv_k);
        %% %        decision_values_mean_video(k,:) = mean(dv_k);
        %% %
        %% %        %decision_values_max_video(k,:) = max(max(dv_k), decision_values_video(k,:));
        %% %        %decision_values_min_video(k,:) = min(min(dv_k), decision_values_video(k,:));
        %% %        %decision_values_median_video(k,:) = median([dv_k ; decision_values_video(k,:)]);
        %% %        %decision_values_mean_video(k,:) = mean([dv_k ; decision_values_video(k,:)]  );
        %% %
        %% %        %decision_values_max_video(k,:) = decision_values_video(k,:);
        %% %        %decision_values_min_video(k,:) = decision_values_video(k,:);
        %% %        %decision_values_median_video(k,:) = decision_values_video(k,:);
        %% %        %decision_values_mean_video(k,:) = decision_values_video(k,:);
        %% %    end
        %% 
        %% 
        %%    classifier.out.probability_estimates = softmax_pr(classifier.out.decision_values);
        %%    classifier.out.pred_pos = sum(classifier.out.decision_values > 0);
        %%    classifier.out.actual_pos = sum(classifier.out.testing_labels_vector_video == 1);
        %%    fprintf('Predicted Number of positives: %d\n', classifier.out.pred_pos);
        %%    fprintf('True number of positives: %d\n', classifier.out.actual_pos);
        %%    disp(size(classifier.out.testing_labels_vector_video));
        %%    disp(size(classifier.out.decision_values));
           % Disabled to check baseline
        %   [classifier.out.recall, classifier.out.precision, classifier.out.ap_info] = vl_pr(classifier.out.testing_labels_vector_video, classifier.out.decision_values);


end
end
