function filenames = listFiles(dirPath, ext)
  %LISTDIRECTORY Lists all files in the `dirPath` that have the file extension `ext`
  stimuliSubfolders = dir(dirPath);
  filenames = {stimuliSubfolders.name};
  filenames = filenames(cellfun(@(x) ~isempty(x), regexp(filenames, ext)));
end

