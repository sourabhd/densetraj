function lse = logsumexp(x)
    lse = log(sum(exp(x)));
end
