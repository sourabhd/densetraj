function [model, predicted_label, accuracy, decision_values, probability_estimates, ...
        recall, precision, ap_info, testing_labels_fname, testing_labels_vector, ...
        pred_pos, actual_pos, loocvscore] = ...
        run_thread_classifier(dset_dir, base_dir, cl, tr, te, tr_v, te_v, fileorder, ...
        retain_frac_threads) 

    t1 = tic;

    addpath('/nfs/bigeye/sdaptardar/installs/libsvm/matlab')
    run('/nfs/bigeye/sdaptardar/installs/vlfeat/toolbox/vl_setup')
    which svmtrain
    which svmpredict
    fileorder.train_files
    fileorder.test_files
    size(tr.train_fv)
    size(te.test_fv)
    size(tr_v.train_fv)
    size(te_v.test_fv)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Parse labels file for training set

    labels_dict_file_train = sprintf('%s%s%s%s%s%s%s%s', dset_dir, '/', 'ClipSets', '/', cl, '_', 'train', '.txt');
    fprintf('%s\n', labels_dict_file_train); 
    [training_labels_fname_video, training_labels_vector_video] = textread(labels_dict_file_train, '%s %d');
    tr_sz = size(tr.train_fv);
    num_tr = tr_sz(1);
    tr_fname = {};
    training_labels_vector = zeros(num_tr,1);
    for i = 1:num_tr
        tr_fname{i} = char(training_labels_fname_video(tr.tr_r2t(i).fileNum,:));
        training_labels_vector(i,1) = training_labels_vector_video(tr.tr_r2t(i).fileNum, :);
    end
    training_labels_fname = char(tr_fname);


    % Parse labels file for testing set
    labels_dict_file_test = sprintf('%s%s%s%s%s%s%s%s', dset_dir, '/', 'ClipSets', '/', cl, '_', 'test', '.txt');
    fprintf('%s\n', labels_dict_file_test); 
    [testing_labels_fname_video, testing_labels_vector_video] = textread(labels_dict_file_test, '%s %d');
    te_sz = size(te.test_fv);
    num_te = te_sz(1);
    te_fname = {};
    testing_labels_vector = zeros(num_te,1);
    for i = 1:num_te
        te_fname{i} = char(testing_labels_fname_video(te.te_r2t(i).fileNum,:));
        testing_labels_vector(i,1) = testing_labels_vector_video(te.te_r2t(i).fileNum, :);
    end
    testing_labels_fname = char(te_fname);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Train a video level classifier

    num_samples_video = size(training_labels_vector_video, 1);
    num_tr_video = size(training_labels_vector_video, 1);
    num_te_video = size(testing_labels_vector_video, 1);
    weight_neg_video = sum(training_labels_vector_video == 1);   % inverse ratio of num of +ve and -ve
    weight_pos_video = sum(training_labels_vector_video == -1);

    opt_video = sprintf('-c 1000 -w1 %f -w-1 %f -t 4', weight_pos_video, weight_neg_video);

    t_kern_start_video = tic;
    Linear_K_video = tr_v.train_fv * tr_v.train_fv';
    t_kern_elapsed_video = toc(t_kern_start_video);
    fprintf('Kernel comp (video): %f sec\n', t_kern_elapsed_video);

    t_cv_start_video = tic;
    n_fold_video = 3;
    % Use AUC as criteria for validation and tuning hyper parameters
    valid_function = @(dec, labels) auc(dec, labels);  % also check validation_function.m in libsvm 
    cv_video = do_binary_cross_validation(training_labels_vector_video, [ (1:num_tr_video)' Linear_K_video], opt_video, n_fold_video); 
    t_cv_elapsed_video = toc(t_cv_start_video);
    fprintf('Baseline AUC (video): %10f\t time = %10f \n', cv_video, t_cv_elapsed_video);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Map thread numbers for lookup

    r_counter = 1;
    t_keyset = {};
    t_valueset = 1:num_tr;
    for i = 1:num_tr_video
        num_tr_vid_i = length(tr.tr_threads_with_fv{i});
        for j = 1:num_tr_vid_i
            t_key = sprintf('%8d_%8d', i, tr.tr_threads_with_fv{i}{j});
            t_keyset{end+1} = t_key;
            r_counter = r_counter + 1;
        end
    end
    thread2row_map = containers.Map(t_keyset,t_valueset);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Some declarations

    n_fold = 3;
    num_tr_pos_video = sum(training_labels_vector_video == 1);
    num_tr_pos = sum(training_labels_vector == 1);
    num_tr_neg_video = sum(training_labels_vector_video == -1);
    num_tr_neg = sum(training_labels_vector == -1);
    loocvscore = zeros(num_tr_pos,1);
    cv_auc = zeros(num_tr_pos,1);
    num_pos_reduced = 0;                                       % fraction to retain
    num_neg_reduced = num_tr_neg;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find LOOCV thread for every thread

    Linear_KK = tr.train_fv * tr_v.train_fv';   % kernel for validation
    weight_neg = num_tr_pos_video-1;
    weight_pos = num_tr_neg_video-1;
    opt = sprintf('-c 1000 -w1 %f -w-1 %f -t 4', weight_pos, weight_neg);
    tr_pos_counter = 1;
    tr_ix_pruned = zeros(0,1);
    num_pos_reduced = 0;
    for v = 1:num_tr_video                                  % for every thread
        if training_labels_vector_video(v,:) == 1                                                     % if +ve
            num_tr_vid_v = length(tr.tr_threads_with_fv{v});
            K = Linear_K_video;                                                                       % drop the video fv
            K(v,:) = [];
            K(:,v) = [];
            L = training_labels_vector_video;                                                         % and labels
            L(v,:) = [];
            cv_auc(tr_pos_counter,1) = do_binary_cross_validation(L, [ (1:num_tr_video-1)' K ], opt, n_fold);
            fprintf('%15s : cv_auc : %5d : %6f\n', cl, v, cv_auc(tr_pos_counter,1));
            model_leaveoneout = svmtrain(L, [ (1:num_tr_video-1)' K], opt); 
            val_ix = zeros(num_tr_vid_v,1);
            for j = 1:num_tr_vid_v
                t = tr.tr_threads_with_fv{v}{j};
                thread_key = sprintf('%8d_%8d', v, t);
                val_ix(j,:) = thread2row_map(thread_key);
            end
            KK = Linear_KK(val_ix,:);
            KK(:,v) = [];
            [pr_lbl, acc, dec_val] = svmpredict(ones(num_tr_vid_v,1), [ (1:num_tr_vid_v)' KK], model_leaveoneout); 
            disp(dec_val);
            [dec_val_sorted, dec_val_sorted_ix] = sort(dec_val, 'descend'); % sort +ve threads by loocv score ranks 
            n = max(round(retain_frac_threads * num_tr_vid_v), 1); 
            num_pos_reduced = num_pos_reduced + n;
            sorted_val_ix = val_ix(dec_val_sorted_ix,:);
            tr_ix_pruned = [  tr_ix_pruned ; sorted_val_ix(1:n,:)  ]; 
            tr_pos_counter = tr_pos_counter + 1;
        end
    end
    tr_ix_pruned = [ tr_ix_pruned ; find(training_labels_vector == -1) ];
    num_neg_reduced = num_tr_neg;
    num_tr_pruned = size(tr_ix_pruned,1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Build model with irrelevant threads pruned

    opt2 = sprintf('-c 1000 -w1 %f -w-1 %f -t 4', num_neg_reduced, num_pos_reduced);  % inverse ratio of num of +ve and -ve samples
    Linear_K_pruned  = tr.train_fv(tr_ix_pruned,:) * tr.train_fv(tr_ix_pruned,:)';
    Linear_KK_pruned = te.test_fv * tr.train_fv(tr_ix_pruned,:)';
    %Linear_K_pruned  = tr_v.train_fv * tr_v.train_fv';
    %Linear_KK_pruned = te.test_fv * tr_v.train_fv';
    %Linear_KK_pruned_video = te_v.test_fv * tr_v.train_fv';
    size(Linear_K_pruned)
    disp(num_tr_pruned)
    size((1:num_tr_pruned)')
    model = svmtrain(training_labels_vector(tr_ix_pruned,:), [ (1:num_tr_pruned)' Linear_K_pruned], opt2); 
    %model = svmtrain(training_labels_vector_video, [ (1:num_tr_video)' Linear_K_pruned], opt2); 

    feat_dim = size(tr.train_fv,2);
    num_test_video = size(te.te_name,1);
    thread_based_features_video = zeros(num_test_video,feat_dim);
    for k = 1:num_test_video
        tfv = zeros(1, feat_dim);
        for l = 1:length(te.te_t2r_hmap{k})
            tfv = tfv + te.test_fv(te.te_t2r_hmap{k}{l}(1), :);
        end
        thread_based_features_video(k,:) = tfv / norm(tfv);
    end
    Linear_KK_tbf_video = thread_based_features_video * tr.train_fv(tr_ix_pruned,:)';
    disp(size(Linear_KK_tbf_video));
    [predicted_label, accuracy, decision_values] = svmpredict(testing_labels_vector_video, [ (1:num_te_video)' Linear_KK_tbf_video], model); 
    %[predicted_label, accuracy, decision_values] = svmpredict(testing_labels_vector, [ (1:num_te)' Linear_KK_pruned], model); 
    %[predicted_label_video, accuracy_video, decision_values_video] = svmpredict(testing_labels_vector_video, [ (1:num_te_video)' Linear_KK_pruned_video], model); 


%    num_test_video = size(te.te_name,1);
%    decision_values_max_video = zeros(num_test_video,1);
%    decision_values_min_video = zeros(num_test_video,1);
%    decision_values_mean_video = zeros(num_test_video,1);
%    decision_values_median_video = zeros(num_test_video,1);
%
%    for k = 1:num_test_video
%        dv_k = zeros(length(te.te_t2r_hmap{k}),1);
%        for l = 1:length(te.te_t2r_hmap{k})
%            dv_k(l,:) = decision_values(te.te_t2r_hmap{k}{l}(1), :);
%        end
%        decision_values_max_video(k,:) = max(dv_k);
%        decision_values_min_video(k,:) = min(dv_k);
%        decision_values_median_video(k,:) = median(dv_k);
%        decision_values_mean_video(k,:) = mean(dv_k);
%
%        %decision_values_max_video(k,:) = max(max(dv_k), decision_values_video(k,:));
%        %decision_values_min_video(k,:) = min(min(dv_k), decision_values_video(k,:));
%        %decision_values_median_video(k,:) = median([dv_k ; decision_values_video(k,:)]);
%        %decision_values_mean_video(k,:) = mean([dv_k ; decision_values_video(k,:)]  );
%
%        %decision_values_max_video(k,:) = decision_values_video(k,:);
%        %decision_values_min_video(k,:) = decision_values_video(k,:);
%        %decision_values_median_video(k,:) = decision_values_video(k,:);
%        %decision_values_mean_video(k,:) = decision_values_video(k,:);
%    end


   probability_estimates = softmax_pr(decision_values);
   pred_pos = sum(decision_values > 0);
   actual_pos = sum(testing_labels_vector_video == 1);
   fprintf('Predicted Number of positives: %d\n', pred_pos);
   fprintf('True number of positives: %d\n', actual_pos);
   disp(size(testing_labels_vector_video));
   disp(size(decision_values));
   [recall, precision, ap_info] = vl_pr(testing_labels_vector_video, decision_values);

    t2 = toc(t1);
    fprintf('Time for : %s : %f sec\n', cl, t2);
end
