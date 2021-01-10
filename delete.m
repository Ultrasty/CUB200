fileFolder=fullfile('./CT');
dirOutput=dir(fullfile(fileFolder,'*/*/._*.png'));
fileNames={dirOutput.name};
fileFolder={dirOutput.folder};

for i = 1:length(fileNames)
      delete([char(fileFolder(i)),'\',char(fileNames(i))])
end
