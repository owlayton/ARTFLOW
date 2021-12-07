classdef HDecoderNet < handle
  %HDecoderNet Multi-output convolutional neural network that decodes heading from MSTd activity.

  properties
    io_config
    decode_params
    nonInputLayers
    outputLayers
    lGraph
    normStats
  end

  methods
    function obj = HDecoderNet(nameValueArgs)
      %MODELSIMULATION Construct an instance of this class
      arguments
        nameValueArgs.io_config struct = readSettings()
        nameValueArgs.decode_params struct = readSettings('decoder.json')
      end

      obj.io_config = nameValueArgs.io_config;
      obj.decode_params = nameValueArgs.decode_params;

      [obj.nonInputLayers, obj.outputLayers] = obj.buildNet();
    end

    function [ds, val_x, val_y, numFeats] = assembleInputs(obj, mode)
      % Load data sample array
      samps = obj.loadSamples(mode);
      % Standardize features
      samps = obj.normalizeFeats(samps, mode);
      % Load label array
      labels = obj.loadLabels(mode);
      % Standardize labels
      labels = obj.normalizeLabels(labels, mode);
      % Record number of input features
      numFeats = size(samps, 1);

      if obj.decode_params.plot.features
        figure();
        plot(samps)
        title([mode, ' features']);
        pause;
      end

      % Creating validation set split from samples and labels
      if strcmpi(mode, 'train')
        valSz = obj.decode_params.train.validation.split_sz;
        [train_x, labels, val_x, val_y] = trainValidationSplit(samps, labels, valSz);

        % Reshape to be consistent with image SSCB format
        samps = reshape(train_x, [size(train_x, 1), 1, 1, size(train_x, 2)]);
        val_x = reshape(val_x, [size(val_x, 1), 1, 1, size(val_x, 2)]);
        % Convert validation set to dlarray
        val_x = dlarray(val_x, 'SSCB');
      else
        samps = reshape(samps, [size(samps, 1), 1, 1, size(samps, 2)]);
        val_x = [];
        val_y = [];
      end

      % Convert samples to Datastore
      sample_ds = arrayDatastore(samps, 'IterationDimension', 4);
      % Convert labels to Datastore
      label_ds = arrayDatastore(labels);

      % Concatenate the sample and label datastores
      ds = combine(sample_ds, label_ds);
    end

    function feats = loadSamples(obj, mode)
      % Get path to the samples applicable for the current mode
      dataDir = getResultsPath(mode);
      % Load the samples as a single struct that holds the features for each sample
      load(fullfile(dataDir, obj.decode_params.io.features_filename), 'act');
      % Convert the struct to a rectangular array: M x N
      feats = cell2mat(actStruct2array(act)')';
    end

    function labels = loadLabels(obj, mode)
      % Get path to the labels applicable for the current mode
      labelDir = getExpPath(mode);
      % Load the samples as a single struct that holds the features for each sample
      labels = table2array(readtable(fullfile(labelDir, 'labels.csv')));
    end

    function samps = normalizeFeats(obj, samps, mode)
      if strcmpi(mode, 'train')
        obj.normStats.x.mu = mean(samps, 2);
        obj.normStats.x.std = std(samps, [],  2);
      end

      % Standardize features
      samps = (samps - obj.normStats.x.mu) ./ obj.normStats.x.std;
    end

    function labels = normalizeLabels(obj, labels, mode)
      if strcmpi(mode, 'train')
        obj.normStats.y.mu = mean(labels, 1);
        obj.normStats.y.std = std(labels, [], 1);
      end

      % Standardize features
      labels = (labels - obj.normStats.y.mu) ./ obj.normStats.y.std;
    end

    function [layers, outLayers] = buildNet(obj)
      arch = obj.decode_params.arch;

      % Create dense/fc layers
      dense_layers = [];
      for i = 1:arch.dense.num
        dense_layers = [dense_layers, ...
          fullyConnectedLayer(arch.dense.units(i), 'Name', sprintf('fc_%d', i))];
      end

      % Create activation function objects
      act_funs = [];
      for i = 1:arch.act.num
        if strcmpi(arch.act.type(i), 'relu')
          currAct = reluLayer('Name', sprintf('relu_%d', i));
        elseif strcmpi(arch.act.type(i), 'sigmoid')
          currAct = sigmoidLayer('Name', sprintf('sigmoid_%d', i));
        elseif strcmpi(arch.act.type(i), 'identity')
           currAct = functionLayer(@(X) X, 'Name', sprintf('identity_%d', i));
        end

        act_funs = [act_funs, currAct];
      end

      % Determine the depth
      stackDepth = min(arch.dense.num, arch.act.num);

      % Assemble the network
      layers = [];
      for i = 1:stackDepth
        layers = [layers, dense_layers(i), act_funs(i)];
      end

      % Append any extra layers without activation funs
      for i = stackDepth+1:arch.dense.num
        layers = [layers, dense_layers(i)];
      end

      % Output layers
      outLayers = [];
      for i = 1:arch.num_outputs
        outLayers = [outLayers, ...
          fullyConnectedLayer(1, 'Name', sprintf('out_%d', i))];
      end
    end

    function [gradients, state, loss] = modelGradients(obj, dlnet, dlX, T1, T2)
      outputLayerNames = obj.getOutputLayerNames();
      [dlY1, dlY2, state] = forward(dlnet, dlX, 'Outputs', outputLayerNames);

      lossY1 = mse(dlY1, T1);
      lossY2 = mse(dlY2, T2);

      loss = sqrt(lossY1.^2 + lossY2.^2);
      gradients = dlgradient(loss, dlnet.Learnables);
    end

    function dlnet = train(obj)
      % Assemble samples and labels for the current mode
      [train_ds, val_x, val_y, numFeats] = obj.assembleInputs('train');

      % Create input layer
      inputLayer = imageInputLayer([numFeats, 1, 1], 'Name', 'input', 'Normalization', 'none');
      % Make netwok graph
      obj.lGraph = layerGraph([inputLayer, obj.nonInputLayers]);

      for o = 1:numel(obj.outputLayers)
        % Connect output layers to rest of network
        obj.lGraph = addLayers(obj.lGraph, obj.outputLayers(o));

        if ~isempty(obj.nonInputLayers)
          obj.lGraph = connectLayers(obj.lGraph, obj.nonInputLayers(end).Name, obj.outputLayers(o).Name);
        else
          obj.lGraph = connectLayers(obj.lGraph, inputLayer.Name, obj.outputLayers(o).Name);
        end
      end

      if obj.decode_params.plot.network_graph
        figure();
        plot(obj.lGraph)
        pause;
      end

      % Create dlnetwork object from net graph
      dlnet = dlnetwork(obj.lGraph);

      % Plot training progress?
      if obj.decode_params.plot.train
        plots = "training-progress";
      else
        plots = "";
      end

      if plots == "training-progress"
        figure();
        lineLossTrain = animatedline('Color', [0.85 0.325 0.098]);
        lineValLossTrain = animatedline('Color', [0.325 0.85 0.098]);
        ylim([0 inf])
        xlabel("Iteration")
        ylabel("Loss")
        grid on
      end

      % Convenience variables
      numEpochs = obj.decode_params.train.epochs;
      miniBatchSz = obj.decode_params.train.mini_batch_sz;
      valFreq = obj.decode_params.train.validation.frequency;
      valPatience = obj.decode_params.train.validation.early_stopping.patience;
      do_early_stopping = obj.decode_params.train.validation.early_stopping.do;

      % Make minibatch queue
      numOutputs = 3;
      mbq = minibatchqueue(train_ds, numOutputs, ...
        'MiniBatchSize', miniBatchSz,...
        'MiniBatchFcn', @preprocessData,...
        'MiniBatchFormat', {'SSCB','',''});

      % Initialize training loop variables
      trailingAvg = [];
      trailingAvgSq = [];
      iteration = 0;

      % Initialize early stopping (if using)
      earlyStop = false;
      if isfinite(valPatience)
        valLosses = inf(1, valPatience);
      end

      % Main training loop
      start = tic;
      for epoch = 1:numEpochs
        % Shuffle data every epoch
        shuffle(mbq);

        if earlyStop
          fprintf('Stopping early on epoch %d\n', epoch);
          break;
        end

        % Loop over mini batches
        while hasdata(mbq) && ~earlyStop
          iteration = iteration + 1;

          [dlX, dlY1, dlY2] = next(mbq);

          % Evaluate the model gradients, state, and loss using dlfeval and the
          % modelGradients function.
          [gradients, state, loss] = dlfeval(@obj.modelGradients, dlnet, dlX, dlY1, dlY2);
          dlnet.State = state;

          % Update the network parameters using the Adam optimizer.
          [dlnet, trailingAvg, trailingAvgSq] = adamupdate(dlnet, gradients, ...
            trailingAvg, trailingAvgSq, iteration);

          % Check validation set accuracy/loss
          if iteration == 1 || mod(iteration, valFreq) == 0
            [y1Pred, y2Pred] = predict(dlnet, val_x);
            valLoss = sqrt(mse(y1Pred, val_y(:, 1)').^2 + mse(y2Pred, val_y(:, 2)').^2);

            if do_early_stopping
              if isfinite(valPatience)
                valLosses = [valLosses valLoss];
                if min(valLosses) == valLosses(1)
                  earlyStop = true;
                else
                  valLosses(1) = [];
                end
              end
            end
          end

          % Display the training progress.
          if plots == "training-progress"
            D = duration(0,0,toc(start),'Format','hh:mm:ss');
            addpoints(lineLossTrain, iteration, double(gather(extractdata(loss))));
            addpoints(lineValLossTrain, iteration, double(gather(extractdata(valLoss))));
            title("Epoch: " + epoch + ", Elapsed: " + string(D));
            drawnow;
          end
        end
      end

      disp('***Finished training network!***');
    end

    function predStruct = predict(obj, dlnet)
      % Assemble samples and labels for the current mode
      [test_ds, ~, ~, ~] = obj.assembleInputs('test');

      % Make minibatch queue
      numOutputs = 3;
      miniBatchSz = obj.decode_params.train.mini_batch_sz;
      mbq = minibatchqueue(test_ds, numOutputs, ...
        'MiniBatchSize', miniBatchSz,...
        'MiniBatchFcn', @preprocessData,...
        'MiniBatchFormat', {'SSCB','',''});

      predsX = [];
      predsY = [];

      % Loop over mini-batches.
      while hasdata(mbq)
        % Read mini-batch of data.
        [dlXTest, dlY1Test, dlY2Test] = next(mbq);

        % Make predictions using the predict function.
        outputLayerNames = obj.getOutputLayerNames();
        [dlY1Pred, dlY2Pred] = predict(dlnet, dlXTest, 'Outputs', outputLayerNames);

        % Record predictions
        predsX = [predsX, dlY1Pred];
        predsY = [predsY, dlY2Pred];
      end

      % Flatten
      predsX = extractdata(predsX(:));
      predsY = extractdata(predsY(:));

      % Unstandardize predictions
      predsX = predsX*obj.normStats.y.std(1) + obj.normStats.y.mu(1);
      predsY = predsY*obj.normStats.y.std(2) + obj.normStats.y.mu(2);

      % Get true labels
      trueLabels = cat(2, cell2mat(readall(test_ds.UnderlyingDatastores{2})));
      % Unstandardize labels
      trueLabels(:, 1) = trueLabels(:, 1)*obj.normStats.y.std(1) + obj.normStats.y.mu(1);
      trueLabels(:, 2) = trueLabels(:, 2)*obj.normStats.y.std(2) + obj.normStats.y.mu(2);

      % Performance metrics
      mse_test = sqrt(msse(trueLabels(:, 1), predsX)^2 + msse(trueLabels(:, 2), predsY)^2);
      mae_test = mean(abs(trueLabels(:, 1) - predsX)) + mean(abs(trueLabels(:, 2) - predsY));

      fprintf('Test set performance summary: MSE = %.2f, MAE = %.2f\n', mse_test, mae_test);

      % Package output struct
      predStruct = struct();
      predStruct.trueX = trueLabels(:, 1);
      predStruct.trueY = trueLabels(:, 2);
      predStruct.predsX = predsX;
      predStruct.predsY = predsY;
      predStruct.mse = mse_test;
      predStruct.mae = mae_test;
    end

    function outputLayerNames = getOutputLayerNames(obj)
      outputLayerNames = strings(obj.decode_params.arch.num_outputs, 1);
      for i = 1:numel(outputLayerNames)
        outputLayerNames(i) = sprintf("out_%d", i);
      end
    end
  end
end

function [X, Y1, Y2] = preprocessData(XCell, YCell)
  % Extract signal data from cell and concatenate
  X = cat(4, XCell{:});
  % Extract label data from cell and concatenate
  Y = cat(1, YCell{:})';

  % Unpack labels
  Y1 = Y(1, :);
  Y2 = Y(2, :);
end

function [train_x, train_y, val_x, val_y] = trainValidationSplit(data, labels, valSz)
  nSamples = size(labels, 1);
  rInds = randperm(nSamples);
  valsetSz = round(valSz*nSamples);
  valInds = rInds(1:valsetSz);
  trainInds = rInds(valsetSz+1:end);

  % Validation samples
  val_x = data(:, valInds);
  % Training samples
  train_x = data(:, trainInds);
  % Validation labels
  val_y = labels(valInds, :);
  % Training labels
  train_y = labels(trainInds, :);
end

function result = msse(true_y, pred_y)
  result = mean((true_y - pred_y).^2);
end
