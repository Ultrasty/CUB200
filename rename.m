fileFolder=fullfile('./CT');

dirOutput=dir(fullfile(fileFolder,'*/*.png'));

fileNames={dirOutput.name};
fileFolder={dirOutput.folder};


for i = 1:length(fileNames)
    str=char(fileNames(i));
    movefile([char(fileFolder(i)),'\',str(1:length(str)-4),'.png'],[char(fileFolder(i)),'\',str(1:length(str)-4),'.jpg'])
end
