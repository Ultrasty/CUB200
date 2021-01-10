
% 把一通道图改成三通道图
fileFolder=fullfile('D:\jupyter\DIP-proj\test2\test');

dirOutput=dir(fullfile(fileFolder,'*.jpg'));

fileNames={dirOutput.name};

for i = 1:length(fileNames)
    path = ['./test2/test/',char(fileNames(i))];
    img = imread(path);
    [X,Y,Z]=size(img) ;
    img_new = zeros(X,Y,3);
    img_new(:,:,1)=img;
    img_new(:,:,2)=img;
    img_new(:,:,3)=img;
    img_new = uint8(img_new);
%   imshow(img_new)
    imwrite(img_new,path)
end