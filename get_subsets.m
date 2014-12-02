function S = get_subsets(A, size_ub)
% get all subsets of A, which have atleast min_cardinality elements
    S = {};
    N = length(A);
    min_cardinality = max(1, N-size_ub+1);
    for i = N:-1:min_cardinality
        S{end+1} = nchoosek(A,i);
    end     
end
