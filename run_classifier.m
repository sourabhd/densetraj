function [model, predicted_label, accuracy, probability_estimates, ...
        recall, precision, ap_info, testing_labels_fname, testing_labels_vector] = ...
        run_classifier(dset_dir, base_dir, cl, train_fv, test_fv, fileorder) 
    addpath('/nfs/bigeye/sdaptardar/installs/liblinear-1.94/matlab')
    run('/nfs/bigeye/sdaptardar/installs/vlfeat/toolbox/vl_setup')
    which train
    which predict
    fileorder.train_files
    fileorder.test_files
    size(train_fv)
    size(test_fv)
    
    labels_dict_file_train = sprintf('%s%s%s%s%s%s%s%s', dset_dir, '/', 'ClipSets', '/', cl, '_', 'train', '.txt');
    fprintf('%s\n', labels_dict_file_train); 
    [training_labels_fname, training_labels_vector] = textread(labels_dict_file_train, '%s %d');
    tr_sz = size(training_labels_fname);
    num_tr = tr_sz(1);

    labels_dict_file_test = sprintf('%s%s%s%s%s%s%s%s', dset_dir, '/', 'ClipSets', '/', cl, '_', 'test', '.txt');
    fprintf('%s\n', labels_dict_file_test); 
    [testing_labels_fname, testing_labels_vector] = textread(labels_dict_file_test, '%s %d');
    te_sz = size(testing_labels_fname);
    num_te = te_sz(1);

    % fileorder_train = cell(num_tr, 1);
    % for i = 1:num_tr
    %    fileorder_train{i} = fileorder.train_files(i).name;
    % end
    % [labels_fname, fileorder_train]
    
    % cv = train(training_labels_vector, sparse(train_fv), '-v 5'); 
    % model = train(training_labels_vector, sparse(train_fv)); 
    %[predicted_label, accuracy, decision_values] = predict(testing_labels_vector, sparse(test_fv), model); 


    % Use AUC as criteria for validation and tuning hyper parameters
    n_fold = 3;
    bestcv = 0;
    for log2c = -5:5,
        cmd = ['-c ', num2str(2^log2c), ' '];
        cv = do_binary_cross_validation(training_labels_vector, sparse(train_fv), cmd, n_fold); 
        if (cv >= bestcv),
          bestcv = cv; bestc = 2^log2c;
        end
        fprintf('%g %g (best c=%g, rate=%g)\n', log2c, cv, bestc, bestcv);
    end 

    best_cmd = ['-c ', num2str(bestc), ' '];
    model = train(training_labels_vector, sparse(train_fv), best_cmd ); 

    
    [predicted_label, accuracy, decision_values] = do_binary_predict(testing_labels_vector, sparse(test_fv), model); 
    probability_estimates = softmax_pr(decision_values);

    fprintf('Predicted Number of positives: %d\n', sum(predicted_label == 1));
    fprintf('True number of positives: %d\n', sum(testing_labels_vector == 1));
    %disp(predicted_label)
    %fprintf('Accuracy: %f', accuracy);
    %disp(probability_estimates);
    [recall, precision, ap_info] = vl_pr(testing_labels_vector, probability_estimates);
end
