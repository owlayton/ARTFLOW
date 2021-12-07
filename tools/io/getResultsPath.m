function expPath = getResultsPath(mode)
  %GETEXPPATH Uses the info in the config file to build the path to the experiment folder where the stimuli are
  %
  % We handle 'simulate' mode as follows:
  % if we have a train subfolder, use that. Otherwise, check for test. Otherwise work in current folder

  arguments
    mode char {mustBeText}
  end
  
  expPath = 'output';
  if inStr('train', mode)
    expPath = fullfile(expPath, 'train');
  else
    expPath = fullfile(expPath, 'test');
  end
end

