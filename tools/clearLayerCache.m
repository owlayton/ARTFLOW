function clearLayerCache(exportPath, keyword)
  matFiles = listFiles(exportPath, '.mat');
  
  % Find all MAT files that have the desired keyword for deleting in it
  mask = cellfun(@(filename) ~isempty(regexpi(filename, keyword, 'once')), matFiles, 'UniformOutput', false);
  relFiles = matFiles(cell2mat(mask));
  
  % Delete the files
  for i = 1:numel(relFiles)
    delete(fullfile(exportPath, relFiles{i}));
  end
end