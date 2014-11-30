function mean_ap = calc_mean_ap(classes, lssvm) 
    fprintf('\n');
    mean_ap = 0;
    num_classes = length(classes);
    for i = 1:num_classes
        fprintf('\n| %20s | %10f |',  classes{i}, lssvm{i}.ap_info.ap);
        mean_ap = mean_ap + lssvm{i}.ap_info.ap;
    end
    mean_ap = mean_ap / num_classes;
    fprintf('\n');
    fprintf('Mean AP : %f\n', mean_ap);
end
