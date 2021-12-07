function runModes = getDecoderRunModes(config)
  %getDecoderRunModes Returns list of modes with which to run the model.
  % Supported modes:
  %  Train: Train network
  %  Test: Predict samples in test subfolder using trained network
  if nargin < 1
    config = readSettings('decoder.json');
  end
  
  % Set modes
  if any(struct2array(config.modes))
    runModes = {'Train', 'Test'};
    runModes = runModes([config.modes.train, config.modes.test]);
  else
    error('No decoder modes turned on.');
  end  
end