classdef NeuralModel < handle
  %NEURALMODEL
  % mtInputLayer corresponds to the "MT preprocessing layer"
  % mtLayer2 corresponds to the first Fuzzy ART Layer
  % mstdInputLayer corresponds to the second Fuzzy ART Layer ("MSTd layer")

  properties
    config
    params
    exportPath
    mtInputLayer
    mtLayer2
    mstdInputLayer
  end

  methods
    function obj = NeuralModel(nameValueArgs)
      %NEURALMODEL Construct an instance of this class
      arguments
        nameValueArgs.config struct = readSettings()
        nameValueArgs.params struct = readSettings('model_params.json')
      end

      % Set the simulation config
      obj.config = nameValueArgs.config;

      % Set the model parameters
      obj.params = nameValueArgs.params;
    end

    function initialize(obj)
      if isempty(obj.mtInputLayer)
        obj.mtInputLayer = obj.loadCachedLayer(obj.params.mt_input.label);
      end
      if isempty(obj.mtLayer2)
        obj.mtLayer2 = obj.loadCachedLayer(obj.params.mt_layer_2.label, true);
      end
      if isempty(obj.mstdInputLayer)
        obj.mstdInputLayer = obj.loadCachedLayer(obj.params.mstd_input_layer.label, true);
      end

      % We need an MT input layer for all simulations
      if isempty(obj.mtInputLayer)
        obj.addMTInputLayer();
      end
    end

    function hasInputLayer = isInitialized(obj)
      hasInputLayer = ~isempty(obj.mtInputLayer);
    end

    function setConfig(obj, config)
      obj.config = config;
    end

    function setParams(obj, params)
      obj.params = params;

      if ~isempty(obj.mtInputLayer)
        obj.mtInputLayer.setParams(params.('mt_input'));
      end

      if ~isempty(obj.mtLayer2)
        obj.mtLayer2.setParams(params.('mt_layer_2'));
      end

      if ~isempty(obj.mstdInputLayer)
        obj.mstdInputLayer.setParams(params.('mstd_input_layer'));
      end
    end

    function clear(obj)
      obj.mtInputLayer = [];
      obj.mtLayer2 = [];
      obj.mstdInputLayer = [];
    end

    function setExportPath(obj, runMode)
      obj.exportPath = getResultsPath(runMode);
    end

    function clearCaches(obj)
      fprintf('  Clearing the following caches:\n');

      % Layers:
      if obj.config.io.clear_cache
        clearLayer("mtInputLayer");
        clearLayer("mtLayer2");
        clearLayer("mstdInputLayer");
      end

      fprintf('Done!\n');

      function clearLayer(label)
        clearLayerCache(obj.exportPath, label);
        fprintf('    %s\n', label);
      end
    end

    function cacheActivation(obj, results)
      if ~exist(obj.exportPath, "dir")
        mkdir(obj.exportPath);
      end

      actStruct = transposeActStruct(results);

      layerNames = fieldnames(actStruct);
      for n = 1:numel(layerNames)
        % Save activations with top-level sample fieldnames for easy loading from model
        filename = fullfile(obj.exportPath, [layerNames{n}, '_activation.mat']);

        % Save layer activation cache if it doesn't already exist
        if ~exist(filename, 'file')
          act = actStruct.(layerNames{n});
          save(filename, '-struct', 'act');
        end

        % Save activations as struct for easy postprocessing
        filename = fullfile(obj.exportPath, [layerNames{n}, '_all_acts.mat']);

        % Save layer activation cache if it doesn't already exist
        if ~exist(filename, 'file')
          act = actStruct.(layerNames{n});
          save(filename, 'act');
        end
      end
    end

    function addMTInputLayer(obj)
      % Try to load the cached layer
      obj.mtInputLayer = obj.loadCachedLayer(obj.params.mt_input.label);
      % If loaded get out, otherwise build from scratch
      if ~isempty(obj.mtInputLayer)
        fprintf('  Loaded cached MT input layer.\n');
        return
      end

      if obj.config.verbose.modes
        fprintf('  Building MT input layer...');
      end

      % Set random seed for reproduceability
      rng(obj.params.random_seed);
      % Get the expected input dimensions
      dims = obj.params.input.dims;
      % Create the layer
      obj.mtInputLayer = MTInputLayer(params=obj.params.mt_input, odeConfig=obj.params.ode, dims=dims);
      % Cache the layer to disk for faster loading next time
      obj.saveLayer(obj.mtInputLayer.label, 'obj.mtInputLayer');

      if obj.config.verbose.modes
        fprintf('Done!\n');
      end
    end

    function addMTLayer2(obj, inputActs)
      % Try to load the cached layer
      obj.mtLayer2 = obj.loadCachedLayer(obj.params.mt_layer_2.label);
      % If loaded get out, otherwise build from scratch
      if ~isempty(obj.mtLayer2)
        fprintf('  Loaded cached MT layer 2.\n');
        return
      end

      if obj.config.verbose.modes
        disp("***************************************************");
        disp('Starting to train MT layer 2 sectors');
      end

      % Create the layer
      if strcmpi('art', obj.params.mt_layer_2.unit_type)
        obj.mtLayer2 = ARTSectorLayer(params=obj.params.mt_layer_2, inputLayer=obj.mtInputLayer);
      else
        obj.mtLayer2 = HebbSectorLayer(params=obj.params.mt_layer_2, inputLayer=obj.mtInputLayer);
      end

      % Train it using the recorded activations to all training stimuli
      if obj.config.verbose.timeit
        tic;
      end

      obj.mtLayer2.train(inputActs);

      if obj.config.verbose.timeit
        toc;
      end

      % Cache the layer to disk for faster loading next time
      obj.saveLayer(obj.mtLayer2.label, 'obj.mtLayer2');

      if obj.config.verbose.modes
        disp("***************************************************");
        disp('Finished training MT layer 2 sectors');
      end
    end

    function addMSTdTemplateLayer(obj, inputActs)
      % Try to load the cached layer
      obj.mstdInputLayer = obj.loadCachedLayer(obj.params.mstd_input_layer.label);
      % If loaded get out, otherwise build from scratch
      if ~isempty(obj.mstdInputLayer)
        fprintf('  Loaded cached MSTd input layer.\n');
        return
      end

      if obj.config.verbose.modes
        disp("***************************************************");
        disp('Starting to train MSTd templates');
      end

      % Create the layer
      if strcmpi('art', obj.params.mt_layer_2.unit_type)
        obj.mstdInputLayer = ARTSectorLayer(params=obj.params.mstd_input_layer, inputLayer=obj.mtLayer2);
      else
        obj.mstdInputLayer = HebbSectorLayer(params=obj.params.mstd_input_layer, inputLayer=obj.mtLayer2);
      end
        
      % Train it using the recorded activations to all training stimuli
      if obj.config.verbose.timeit
        tic;
      end

      obj.mstdInputLayer.train(inputActs);

      if obj.config.verbose.timeit
        toc;
      end

      % Cache the layer to disk for faster loading next time
      obj.saveLayer(obj.mstdInputLayer.label, 'obj.mstdInputLayer');

      if obj.config.verbose.modes
        disp("***************************************************");
        disp('Finished training MSTd templates');
      end
    end

    function saveLayer(obj, layerLabel, layerObjName)
      % Make folders if they don't exist
      if ~exist(obj.exportPath, 'dir')
        mkdir(obj.exportPath);
      end

      % We can't export obj.var so create a local variable for the layer called 'layer' with the layer data then export
      % that
      layer = eval(layerObjName);
      save(fullfile(obj.exportPath, layerLabel), 'layer');
    end

    function loadedLayer = loadCachedLayer(obj, layerLabel, trainPathOnly)
      if nargin > 2 && trainPathOnly && endsWith(obj.exportPath, 'test', 'IgnoreCase', true)
        exportPath = fullfile(extractBefore(obj.exportPath, 'test'), 'train');
      else
        exportPath = obj.exportPath;
      end

      % We set the layer initially as MTInputLayer only because we need an object type
      loadedLayer = MTInputLayer.empty(1, 0);
      if exist(fullfile(exportPath, [layerLabel, '.mat']), 'file')
        load(fullfile(exportPath, layerLabel), 'layer');
        loadedLayer = layer;
      end
    end

    function act = loadCachedLayerActivation(obj, stimulusName, layerLabel)
      act = MTInputLayer.empty(1, 0);

      filename = fullfile(obj.exportPath, [layerLabel, '_activation.mat']);

      if exist(filename, 'file')
        load(filename, stimulusName);
        act = eval(stimulusName).act;

        if obj.config.verbose.stimuli
          fprintf('    Loaded cached %s activation for %s\n', layerLabel, stimulusName);
        end
      end
    end

    function layerActStruct = simulateSystem(obj, stimulusPath, stimulusName, sceneStruct, obsStruct)
      %simulateSystem Simulate the neural system ODE
      % Make sure config file input dims matches actual flow
      if any(obj.params.input.dims ~= sceneStruct.dims)
        error("Flow dims mismatch. Config: " + obj.params.input.dims + ", Flow mat file: " + sceneStruct.dims);
      end

      % Initialize struct to hold activations in each layer
      layerActStruct = struct();

      % How many frames are we simulating? -1 means simulate all available ones
      nFrames = obj.params.ode.num_frames;
      if nFrames == -1
        nFrames = sceneStruct.numFrames;
      end

      % How often do we readout model activation? Even if we don't export, always get the layer activations on the last
      % frame
      exportFrame = nFrames;

      % Clear neural activation before simulating each sample
      obj.mtInputLayer.clearActivation();

      % Set random seed for reproduceability
      rng(obj.params.random_seed);

      % Load cached MT input activation if it exists
      mtInputLayerAct = obj.loadCachedLayerActivation(stimulusName, obj.mtInputLayer.label);
      if isempty(mtInputLayerAct)
        mtInputLayerAct = simulateMT(nFrames, exportFrame);
      end

      layerActStruct.(obj.mtInputLayer.label).act = mtInputLayerAct;

      % Simulate MT layer 2 (sectors)
      if ~isempty(obj.mtLayer2)
        % Load cached MT Layer 2 activation if it exists
        mtLayer2Act = obj.loadCachedLayerActivation(stimulusName, obj.mtLayer2.label);
        if isempty(mtLayer2Act)
          inputActs = layerActStruct.(obj.mtInputLayer.label).act;
          mtLayer2Act = obj.mtLayer2.predict(inputActs);
        end

        layerActStruct.(obj.mtLayer2.label).act = mtLayer2Act;
      end

      % Simulate MSTd input layer (learned templates)
      if ~isempty(obj.mstdInputLayer)
        % Load cached MT Layer 2 activation if it exists
        mstdInputLayerAct = obj.loadCachedLayerActivation(stimulusName, obj.mstdInputLayer.label);
        if isempty(mstdInputLayerAct)
          inputActs = layerActStruct.(obj.mtLayer2.label).act;

          % Convert uneven features to array
          inputActs = cell2mat(inputActs')';

          mstdInputLayerAct = obj.mstdInputLayer.predict(inputActs);
        end

        layerActStruct.(obj.mstdInputLayer.label).act = mstdInputLayerAct;
      end

      function act = simulateMT(nFrames, exportFrame)
        simParams = struct();
        simParams.num_time_steps = obj.params.ode.num_time_steps;

        for f = 1:nFrames
          currInputStruct = getCurrInput(sceneStruct, obsStruct, f, stimulusPath);
          simParams.t_frame = f;

          for s = 1:obj.params.ode.num_time_steps
            simParams.t_step = s;

            % Simulate the MT input layer
            obj.mtInputLayer.simulate(currInputStruct, simParams);
          end

          % Are we reading out model activation on this frame?
          if f == exportFrame
            act = obj.mtInputLayer.getActivation();
          end
        end
      end
    end
  end
end