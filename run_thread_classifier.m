function [model, predicted_label, accuracy, decision_values, probability_estimates, ...
        recall, precision, ap_info, testing_labels_fname, testing_labels_vector, ...
        pred_pos, actual_pos, loocvscore] = ...
        run_thread_classifier(dset_dir, base_dir, cl, tr, te, fileorder, ...
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

    labels_dict_file_train = sprintf('%s%s%s%s%s%s%s%s', dset_dir, '/', 'ClipSets', '/', cl, '_', 'train', '.txt');
    fprintf('%s\n', labels_dict_file_train); 
    [training_labels_fname_f, training_labels_vector_f] = textread(labels_dict_file_train, '%s %d');
    tr_sz = size(tr.train_fv);
    num_tr = tr_sz(1);
    tr_fname = {};
    training_labels_vector = zeros(num_tr,1);
    for i = 1:num_tr
        tr_fname{i} = char(training_labels_fname_f(tr.tr_r2t(i).fileNum,:));
        training_labels_vector(i,1) = training_labels_vector_f(tr.tr_r2t(i).fileNum, :);
    end
    training_labels_fname = char(tr_fname);


    labels_dict_file_test = sprintf('%s%s%s%s%s%s%s%s', dset_dir, '/', 'ClipSets', '/', cl, '_', 'test', '.txt');
    fprintf('%s\n', labels_dict_file_test); 
    [testing_labels_fname_f, testing_labels_vector_f] = textread(labels_dict_file_test, '%s %d');
    te_sz = size(te.test_fv);
    num_te = te_sz(1);
    te_fname = {};
    testing_labels_vector = zeros(num_te,1);
    for i = 1:num_te
        te_fname{i} = char(testing_labels_fname_f(te.te_r2t(i).fileNum,:));
        testing_labels_vector(i,1) = testing_labels_vector_f(te.te_r2t(i).fileNum, :);
    end
    testing_labels_fname = char(te_fname);

    num_samples = size(training_labels_vector, 1);
    weight_neg = sum(training_labels_vector == 1);   % inverse ratio of num of +ve and -ve
    weight_pos = sum(training_labels_vector == -1);

    opt = sprintf('-c 100 -w1 %f -w-1 %f -t 4', weight_pos, weight_neg);

    % Use precomputed training matrix for training
    t_kern_start = tic;
    Linear_K = tr.train_fv * tr.train_fv';
    t_kern_elapsed = toc(t_kern_start);
    fprintf('Kernel comp: %f sec\n', t_kern_elapsed);

    t_cv_start = tic;
    n_fold = 3;
    % Use AUC as criteria for validation and tuning hyper parameters
    valid_function = @(dec, labels) auc(dec, labels);  % also check validation_function.m in libsvm 
    cv = do_binary_cross_validation(training_labels_vector, [ (1:num_tr)' Linear_K], opt, n_fold); 
    t_cv_elapsed = toc(t_cv_start);
    fprintf('Baseline AUC: %10f\t time = %10f \n', cv, t_cv_elapsed);

    loocvscore = zeros(num_tr,1);

    for j = 1:num_tr
        K = Linear_K;
        K(j,:) = [];
        K(:,j) = [];
        L = training_labels_vector;
        L(j,:) = [];
        loocvscore(j,1) = do_binary_cross_validation(L, [ (1:num_tr-1)' K ], opt, n_fold); 
        loocvscore(j,1) = (cv -  loocvscore(j,1)) / cv;
    end


    num_pos = sum(training_labels_vector == 1);
    num_neg = sum(training_labels_vector == -1);
    num_pos_reduced = int32(retain_frac_threads * num_pos);           % prune irrelevant threads 
    num_neg_reduced = num_neg;
    [tr_sorted, tr_sorted_ix] = sort(loocvscore, 'descend');
    tr_sorted_pos_ix = find(training_labels_vector(tr_sorted_ix,:) == 1);
    tr_sorted_neg_ix = find(training_labels_vector(tr_sorted_ix,:) == -1);
    tr_ix_pruned = [  tr_sorted_pos_ix(1:num_pos_reduced,:) ; tr_sorted_neg_ix(1:num_neg_reduced,:) ]; 
    num_tr_pruned = size(tr_ix_pruned,1);


    opt2 = sprintf('-c 100 -w1 %f -w-1 %f -t 4', num_neg, num_pos);  % inverse ratio of num of +ve and -ve samples
    Linear_K_pruned  = tr.train_fv(tr_ix_pruned,:) * tr.train_fv(tr_ix_pruned,:)';
    Linear_KK_pruned = te.test_fv * tr.train_fv(tr_ix_pruned,:)';
    size(Linear_K_pruned)
    disp(num_tr_pruned)
    size((1:num_tr_pruned)')
    model = svmtrain(training_labels_vector(tr_ix_pruned,:), [ (1:num_tr_pruned)' Linear_K_pruned], opt2); 

    [predicted_label, accuracy, decision_values] = svmpredict(testing_labels_vector, [ (1:num_te)' Linear_KK_pruned], model); 


    num_test_video = size(te.te_name,1);
    decision_values_video = zeros(num_test_video,1);
    for k = 1:num_test_video
        decision_values_video(k,:) = -Inf;
        for l = 1:length(te.te_t2r_hmap{k})
            d = decision_values(te.te_t2r_hmap{k}{l}(1), :);
            if d > decision_values_video(k,:)
                decision_values_video(k,:) = d;
            end
        end
    end


    probability_estimates = softmax_pr(decision_values_video);
    pred_pos = sum(decision_values_video > 0);
    actual_pos = sum(testing_labels_vector_f == 1);
    fprintf('Predicted Number of positives: %d\n', pred_pos);
    fprintf('True number of positives: %d\n', actual_pos);
    [recall, precision, ap_info] = vl_pr(testing_labels_vector_f, decision_values_video);

    t2 = toc(t1);
    fprintf('Time for : %s : %f sec\n', cl, t2);
end
