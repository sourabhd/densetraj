function A = normavg(X, idx)
    A = zeros(size(X,1),1);
    M = mean(X(idx,:),1);
    norm_M = norm(M);
    if norm_M > 0
        A = M / norm_M;
    end
end
