function filenames = listDirectory(dirPath)
  %LISTDIRECTORY Lists all folders in the `dirPath` that might contain stimuli (not train/test/image folders).
  stimuliSubfolders = dir(dirPath);
  folderNames = {stimuliSubfolders.name};
  isFolder = {stimuliSubfolders.isdir};
  isFolder = cell2mat(isFolder);
  filenames = folderNames(isFolder);
  % We dont want to queue up Train/Test/Image folders as stimuli folders
  filenames = filenames(cellfun(@isempty, regexp(filenames, '^\.+|^Train|^Test|^Images')));
end

