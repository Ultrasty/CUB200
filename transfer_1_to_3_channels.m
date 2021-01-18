% 把一通道图改成三通道�?
fileFolder=fullfile('./CT');

dirOutput=dir(fullfile(fileFolder,'*/*.jpg'));

fileNames={dirOutput.name};
fileFolder={dirOutput.folder};

for i = 1:length(fileNames)
    path = [char(fileFolder(i)),'\',char(fileNames(i))];
    img = imread(path);
    [X,Y,Z]=size(img) ;
    if Z==1
        img_new = zeros(X,Y,3);
        img_new(:,:,1)=img;
        img_new(:,:,2)=img;
        img_new(:,:,3)=img;
        img_new = uint8(img_new);
        imwrite(img_new,path)
    end
end
