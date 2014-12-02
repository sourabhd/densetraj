function A = normavg2(cumsumX, idxr)
    D = size(cumsumX, 2);
    A = zeros(1, D);
    %disp(idxr);
    %class(idxr)
    %idx = cell2mat(idxr);
    idx = idxr;
    ids = sort(idx);
    lenIdx = length(idx);

    l = ids(1)-1;
    h = ids(1);
    for i = 1:lenIdx-1
        % fprintf('l=%6d h=%6d\n', l, h);
        if ids(i) + 1 ~= ids(i+1)
            % fprintf('-----------------\n');
            A = A + cumsumX(h+1,:) - cumsumX(l+1,:);
            l = ids(i+1) - 1;
            h = ids(i+1);
        else
            h = h + 1; 
        end
    end
    % fprintf('l=%6d h=%6d\n', l, h);
    % fprintf('-----------------\n');
    A = A + cumsumX(h+1,:) - cumsumX(l+1,:);

    norm_A = norm(A);
    if norm_A > 0
        A = A / norm_A;
    else
        A = zeros(1,D);
    end

end
