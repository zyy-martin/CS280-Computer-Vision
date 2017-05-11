
figure;    
for i = 1 : length(points)
    scatter3(points(i,1),points(i,2),points(i,3),10,'blue')
    hold on
end
camera1 = [0 0 0];
camera2 = -R2^(-1)*t2;
scatter3(camera1(1),camera1(2),camera1(3),10,'black')
hold on
scatter3(camera2(1),camera2(2),camera2(3),10,'red')
title(['3d reconstruction for ',name])
