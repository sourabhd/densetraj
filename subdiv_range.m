function [r_start_o, r_end_o] = subdiv_range(r_start, r_end, a, b, lb)
    n = r_end - r_start + 1;
    if n <= lb
        r_start_o = r_start;
        r_end_o = r_end;
    else
        fprintf('%10d %10d : %10d\n', r_start, r_end, r_end - r_start + 1);
        intvl = floor(n * (b - 1) / (b * (a - 1)));
        for i = 0:a-1
            %[r_start_o, r_end_o] = subdiv_range(r_start + i * floor(n / b), r_start + (i + 1) * floor(n / b) - 1, a, b, lb);
            s = r_start + i * intvl;
            e = s + floor(n / b) - 1;
            [r_start_o, r_end_o] = subdiv_range(s, e, a, b, lb);
        end
    end
end
