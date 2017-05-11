function [R, t] = find_rotation_translation(E)
    % svd
    [U, ~, V] = svd(E, 0);
    
    % t
    t{1} = U(:,end);
    t{2} = -U(:,end);
    
    % R
    rot_mat{1} = [0, -1, 0; 1, 0, 0; 0, 0, 1];
    rot_mat{2} = [0, 1, 0; -1, 0, 0; 0, 0, 1];
    count = 0;
    for i = 1:2
        for j = 1:2            
            temp = power(-1, i) * U * rot_mat{j} * V';
            % filter R by the propoerty that the determinant of a roatation
            % matrix is 1
            if abs(det(temp) - 1) < 1e-5
                count = count + 1;
                R{count} = temp;
            end
        end
    end
end
