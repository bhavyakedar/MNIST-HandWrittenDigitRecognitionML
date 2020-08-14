img = imread('ml_test2_9.jpeg');
img = img(:,:,1);
img = double(img);
im = imresize(img, [28 28]);
im = im / max(max(im));
im = 1-im;

for i = 1:size(im,1)
    for j = 1:size(im,2)
        if im(i,j) < 0.1
            im(i,j) = 0;
        end
        if im(i,j) > 0.9
            im(i,j) = 1;
        end
    end
end

imagesc(im);
colormap(gray);
pause;

im = im';
im = [1 im(:)'];
prediction = all_theta*im';
[~,ind] = max(prediction);
title(['Prediction : ',int2str(mod(ind,10))]);