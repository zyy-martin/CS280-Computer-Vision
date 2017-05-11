function [points_3d, err] = find_3d_points(P1, P2, points)
    N = size(points, 1);
    points_3d = zeros(N, 3);
    err = 0;
    
    for k = 1:N
    % construct A
    A(1,:) = P1(3,:) * points(k,1) - P1(1,:);
    A(2,:) = P1(3,:) * points(k,2) - P1(2,:);
    A(3,:) = P2(3,:) * points(k,3) - P2(1,:);
    A(4,:) = P2(3,:) * points(k,4) - P2(2,:);
    
    [~,~,V] = svd(A, 0);
    homo_pt = V(:, end);
    points_3d(k,:) = homo_pt(1:3)'/homo_pt(4);
    
    proj_pt1 = P1 * homo_pt;
    proj_pt1 = proj_pt1(1:2)/proj_pt1(3);
    proj_pt2 = P2 * homo_pt;
    proj_pt2 = proj_pt2(1:2)/proj_pt2(3);
    
    err = err + (norm(proj_pt1 - points(k, 1:2)') + norm(proj_pt2 - points(k, 3:4)'))/2;
    end
    err = err/N;
end