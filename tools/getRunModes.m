function runModes = getRunModes(config)
  %getRunModes Returns list of modes with which to run the model.
  % Supported modes:
  %  TrainTemplates: simulate model + learn MSTd templates
  %  Test: simulate model + test on test stimuli (assumes learning already completed
  %  Simulate: simulate model without any learning/testing.
  %    Operates on 'train' stimuli (if folder exists), otherwise on 'test' stimuli (if folder exists), otherwise just
  %    runs stimuli in experiment folder.
  if nargin < 1
    config = readSettings();
  end

  % Enumeration of all supported modes
  modeCategories = fieldnames(config.modes);
  trainModes = fieldnames(config.modes.(modeCategories{1}));
  trainModes = cellfun(@(m) [modeCategories{1}, '_', m], trainModes, 'UniformOutput', false);
  simModes = fieldnames(config.modes.(modeCategories{2}));
  simModes = cellfun(@(m) [modeCategories{2}, '_', m], simModes, 'UniformOutput', false);

  % Set modes
  runModes = [];
  if any(struct2array(config.modes.train))
    runModes = [runModes; trainModes(struct2array(config.modes.train))];
  end
  if any(struct2array(config.modes.simulate))
    runModes = [runModes; simModes(struct2array(config.modes.simulate))];
  end

  if isempty(runModes)
    error("No run modes enabled in config file.");
  end
end