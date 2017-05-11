function [F, res_err] = fundamental_matrix(points)
    N = size(points, 1);
    
    % Normalization
    mean_x1 = mean(points(:,1));
    mean_y1 = mean(points(:,2));
    mean_x2 = mean(points(:,3));
    mean_y2 = mean(points(:,4));
    
    factor1 = sqrt(2)/mean(sqrt((points(:,1) - mean_x1).^2 + (points(:,2) - mean_y1).^2));
    factor2 = sqrt(2)/mean(sqrt((points(:,3) -mean_x2).^2 + (points(:,4) - mean_y2).^2));
    
    T1 = [factor1, 0, -factor1*mean_x1; 0, factor1, -factor1*mean_y1; 0, 0, 1];
    T2 = [factor2, 0, -factor2*mean_x2; 0, factor2, -factor2*mean_y2; 0, 0, 1];
    
    loc1 = (T1 * [points(:, 1:2) ones(N, 1)]')';
    loc2 = (T2 * [points(:, 3:4) ones(N, 1)]')';
    
    % Optimization
    A = [loc1(:,1).*loc2(:,1) loc1(:,2).*loc2(:,1) loc2(:,1) loc1(:,1).*loc2(:,2) loc1(:,2).*loc2(:,2) loc2(:,2) loc1(:,1) loc1(:,2) ones(N,1)];
    [~, ~, V] = svd(A, 0);
    F = reshape(V(:,end), [3 3])';
    
    % Approximation for rank 2
    [U,S,V] = svd(F, 0);
    D = diag(S,0);
    F = U * diag([D(1:end-1); 0]) * V';
    
    % Denormalization
    F = T2' * F * T1;
    
    
    res_err = 0;
    for k = 1:N
        proj_x1 = F * [points(k,1:2) 1]';
        %proj_x2 = F * [points(k,3:4) 1]';
        res_err = res_err + abs([points(k,3:4) 1] * proj_x1)/sqrt(proj_x1(1)^2 + proj_x1(2)^2);
        %res_err = res_err + abs([points(k,1:2) 1] * proj_x2)/sqrt(proj_x2(1)^2 + proj_x2(2)^2);
    end
    res_err = res_err/N;
end