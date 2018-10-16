dir_lsun_val = dir('features');

figure('color' , 'white' , 'position' , [1 , 1 , 2000 , 1200]);
for i = 3 : size(dir_lsun_val)
    load(['features/' dir_lsun_val(i).name]);
    feature = permute(feature , [3 , 4 , 2 , 1]);
    feature = feature(: , : , 2 : 4);
    feature = max(feature , [] , 3);
    img = imread(['datasets/lsun/images/' dir_lsun_val(i).name(1 : end - 4) '.png']);
    subplot(121) , imshow(img);
    subplot(122) , imagesc(exp(feature)) , axis equal , axis off;
%     subplot(223) , imagesc(feature(: , : , 3)) , axis equal , axis off;
%     subplot(224) , imagesc(feature(: , : , 4)) , axis equal , axis off;
    ginput(1);
end