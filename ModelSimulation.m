classdef ModelSimulation < handle
  %MODELSIMULATION Simulates a model: train, test, or just simulation modes
  
  properties
    config
    templateModelName
    neuralModel
    currInputPath  % Path to stimuli in the current experiment
    stimuli  % List of stimuli to process
    stimulusActs  % Final-frame activation to each stimulus in queue
  end
  
  methods
    function obj = ModelSimulation(nameValueArgs)
      %MODELSIMULATION Construct an instance of this class
      arguments
        nameValueArgs.config struct = readSettings()
        nameValueArgs.templateModelName char = 'fuzzyART'
      end
      
      % Set the simulation config
      obj.config = nameValueArgs.config;
      
      % Set the model that we are working with to learn templates
      obj.templateModelName = nameValueArgs.templateModelName;
      
      % Initialize neural model
      obj.neuralModel = NeuralModel(config=obj.config);
    end
    
    function results = run(obj)
      % Figure out which run modes we are doing
      runModes = {'train_mtLayer2', 'train_mstdTemplates', 'simulate_all_test'};
      
      % Clear caches
      for m = 1:length(runModes)
        if obj.config.verbose.modes
          disp(['Mode: ', runModes{m}]);
        end
        
        % Set the neural model's export path based on the run mode
        obj.neuralModel.setExportPath(runModes{m});
        
        % Clear the cache in each mode folder
        obj.neuralModel.clearCaches();
      end
      
      % Loop through run modes.
      for m = 1:length(runModes)
        runMode = runModes{m};

        switch runMode
          case 'train_mtLayer2'
            % Simulate model in its current state, then train
            obj.simulate(runMode);
            obj.trainMTLayer2();
          case 'train_mstdTemplates'
            % Simulate model in its current state, then train
            obj.simulate(runMode);
            obj.trainMSTdTemplates();
        end
        
        % Simulate model in its current state
        results = obj.simulate(runMode);
      end

      % Decode the test samples
      runHeadingDecoder(["train", "test"]);
    end
    
    function setInputQueue(obj, runMode)
      %setInputQueue Get a listing of all the folders containing stimuli that will be processed
      obj.currInputPath = getExpPath(runMode);
      obj.stimuli = listDirectory(obj.currInputPath);
      
      % Limit number of stimuli based on config file. -1 means process all stimuli
      if obj.config.io.num_stimuli > 0
        obj.stimuli = obj.stimuli(1:obj.config.io.num_stimuli);
      end
    end
    
    function preprocessOpticFlow(obj)
      %preprocessOpticFlow Runs optic flow detection algorithm on stimuli
      if ~obj.config.preprocessing.do
        return;
      end
    end

    function initialize(obj, runMode)
      % Set the neural model's export path based on the run mode
      obj.neuralModel.setExportPath(runMode);

      % Set the path to where stimuli are and queue up stimuli to process
      obj.setInputQueue(runMode);

      % Check to make sure neural model has been initialized
      obj.neuralModel.initialize();
    end
    
    function results = simulate(obj, runMode)
      obj.initialize(runMode);

      if obj.config.verbose.modes
        disp("***************************************************");
        fprintf('Starting simulation of %d stimuli (%s mode)...\n', numel(obj.stimuli), runMode);
      end
      
      % Determine if we run stimuli in parallel or not
      % We do this if we want to run in parallel and if we are processing all stimuli (not just one)
      if obj.config.verbose.timeit
        tic;
      end

      if obj.config.simulate.parallel
        results = obj.runParallelSimulation();
      else
        results = obj.runSerialSimulation();
      end

      if obj.config.verbose.timeit
          toc;
      end

      if obj.config.verbose.modes
        disp("***************************************************");
        fprintf('Finished simulation of %d stimuli...\n', numel(obj.stimuli));
      end
      
      % Cache layer activations
      obj.cacheActivation(results);
      
      % Build 2D M x N data arrays of activations
      obj.stimulusActs = obj.collateActivations(results);
    end
    
    function results = runParallelSimulation(obj)
      queue = obj.stimuli;
      inputPath = obj.currInputPath;
      
      % Create parallel pool object
      p = gcp('nocreate');
      if isempty(p)
        p = parpool('local', obj.config.simulate.num_parallel_nets);
      end
      
      % Partition stimulus indices across the workers
      q_inds = 1:numel(queue);
      q_inds_dist = distributed(q_inds);
      
      spmd
        results = struct();
        for s = getLocalPart(q_inds_dist)
          % Assumes mat filename matches folder name
          filePath = fullfile(inputPath, queue{s});
          results.(queue{s}) = obj.runSimulation(obj.neuralModel, filePath, queue{s}, s);
        end
      end
      
      % Merge the results distributed across the workers
      results = mergeDistResults(results);
    end
    
    function results = runSerialSimulation(obj)
      queue = obj.stimuli;
      
      results = struct();
      for s = 1:numel(queue)
        % Assumes mat filename matches folder name
        filePath = fullfile(obj.currInputPath, queue{s});
        results.(queue{s}) = obj.runSimulation(obj.neuralModel, filePath, queue{s}, s);
        %         saveas(gcf, [queue{s}, '.png']);
      end
    end
    
    function currStimulusActs = runSimulation(obj, neuralModel, stimulusPath, stimulusName, stimulusNum)
      if obj.config.verbose.stimuli
        fprintf('  Starting to simulate %s (%d)...\n', stimulusName, stimulusNum);
      end

      % Read in the stimulus sample
      [sceneStruct, observerStruct] = loadSample(stimulusPath, stimulusName);
      
      % Simulate the neural system
      currStimulusActs = neuralModel.simulateSystem(stimulusPath, stimulusName, sceneStruct, observerStruct);
      
      if obj.config.verbose.stimuli
        fprintf('    Finished simulation of %s.\n', stimulusName);
      end
    end


    
    function acts = collateActivations(obj, resultsStruct)
      %collateActivations Build struct with fields for the different layer activations over all inputs
      % NOTE: This should probably happen in NeuralModel
      acts = struct();
      
      resultsNames = fieldnames(resultsStruct);
      firstResult = resultsStruct.(resultsNames{1});
      
      if isfield(firstResult, 'mtInputLayer')
        inputActs = actStruct2array(resultsStruct, 'mtInputLayer');
        acts.('mtInputLayer') = inputActs;
      end
      
      if isfield(firstResult, 'mtLayer2')
        inputActs = actStruct2array(resultsStruct, 'mtLayer2');
        acts.('mtLayer2') = inputActs;
      end
      
      if isfield(firstResult, 'mstdInputLayer')
        inputActs = actStruct2array(resultsStruct, 'mstdInputLayer');
        acts.('mstdInputLayer') = inputActs;
      end
    end
    
    function trainMTLayer2(obj)
      % Make an MTLayer2 object, which partitions input space into a regular grid of sectors, and train it
      inputAct = obj.stimulusActs.('mtInputLayer');
      obj.neuralModel.addMTLayer2(inputAct);
    end
    
    function trainMSTdTemplates(obj)
      % Make an MSTd input layer object, which partitions input space into a regular grid of sectors, and train it
      inputAct = obj.stimulusActs.('mtLayer2');
      obj.neuralModel.addMSTdTemplateLayer(inputAct);
    end
    
    function cacheActivation(obj, results)
      obj.neuralModel.cacheActivation(results);
    end
  end
end

function [Env, Obs] = loadSample(stimulusPath, stimulusName)
  % Load in stimulus MAT file (Analytic format)
  if exist(fullfile(stimulusPath, [stimulusName, '.mat']), "file")
    load(fullfile(stimulusPath, stimulusName), 'Env', 'Obs');
    % Fill in extra fields
    Env.fileFormat = 'analytic';
    Env.numFrames = numel(fieldnames(Env.dx));
  else
    % Airsim format (organized by frame — load in first frame now to populate info)
    % Get the frame filenames
    frameFiles = listFiles(fullfile(stimulusPath, 'OpticFlow'), 'mat');
    % Load needed variables
    load(fullfile(stimulusPath, 'OpticFlow', frameFiles{1}), 'dims', 'fileNumDigits');
    % Make dummy Env struct
    Env.fileFormat = 'airsim';
    Env.fileNumDigits = fileNumDigits;
    Env.dims = dims;
    Env.numFrames = numel(frameFiles);
    Env.fps = 30;
    % Make dummy Obs struct
    Obs.fov = 90;
  end
end