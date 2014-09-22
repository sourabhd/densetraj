function y = softmax_pr(x)
    y = exp(x - logsumexp(x));
end
