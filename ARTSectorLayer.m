classdef ARTSectorLayer < TrainableLayer
  %MTInputLayer

  properties
    inputLayer  % Layer that supplies feedforward input for training
    inputSectorLabels  % Maps each cell to a sector int code
    fbSectorLabels  % Labels each input cell to the sector it was in within the previous layer
    numFeats  % Number of cells in each sector
    sectorUnits
    rfCenters2D  % Coordinates of sectors (row, col)
    dims  % Dimensions of sector grid
  end

  methods
    function obj = ARTSectorLayer(nameValueArgs)
      %MTInputLayer Construct an instance of this class
      arguments
        nameValueArgs.params struct
        nameValueArgs.inputLayer Layer
        nameValueArgs.initialize logical = true
      end

      % Create instance vars for layer params and ODE config in a consistent way across layers
      obj = obj@TrainableLayer(params=nameValueArgs.params);

      obj.inputLayer = nameValueArgs.inputLayer;

      % Initialize units in layer
      if nameValueArgs.initialize
        obj.initialize()
      end
    end

    function initialize(obj)
      obj.dims = obj.params.num_sectors;

      if isa(obj.inputLayer, 'ARTSectorLayer')
        % Get number of cells present in each sector
        numCellsPerSector = arrayfun(@(s) s.getNumCommitted(), obj.inputLayer.sectorUnits, 'UniformOutput', false);

        % Get a total of all cells over all sectors
        numInputCells = sum(cell2mat(numCellsPerSector));
        % Compute their RF positions using a linear index cursor
        obj.fbSectorLabels = zeros(numInputCells, 1);
        cursor = 0;
        for s = 1:numel(numCellsPerSector)
          obj.fbSectorLabels(cursor+1:cursor+numCellsPerSector{s}) = s;
          cursor = cursor + numCellsPerSector{s};
        end

        % Determine 2D RF positions of committed neurons in input layer in their grid (we didn't know them before
        % training, only their grid)
        [rf_x, rf_y] = ind2sub(obj.inputLayer.dims, obj.fbSectorLabels);
        rfCenters2D = [rf_x, rf_y];
      else
        % Dynamic (nontrainable) layer
        numInputCells = size(obj.inputLayer.rfCenters2D, 1);
        % Use RF positions directly from previous layer
        rfCenters2D = obj.inputLayer.rfCenters2D;
      end

      % Setup grid RFs
      obj.initializeSectors(numInputCells, rfCenters2D);
      % Initialize Fuzzy ART objects in each sector
      obj.initializeSectorObjects();
    end

    function initializeSectors(obj, numInputCells, rfCenters)

      numSectors = prod(obj.dims);

      % Partition the input space into a grid of sectors
      obj.inputSectorLabels = zeros(numInputCells, 1);
      obj.numFeats = zeros(numSectors, 1);

      % Determine number of rows and columns that define a sector
      sectorSz = round(obj.inputLayer.dims' ./ obj.dims);

      sectorCounter = 1;
      for c = 1:obj.params.num_sectors(2)
        for r = 1:obj.params.num_sectors(1)
          % Get rows in valid sector
          cond = rfCenters(:, 1) >= (r-1)*sectorSz(1)+1 & rfCenters(:, 1) < r*sectorSz(1)+1;
          % Get cols in valid sector
          cond = cond & rfCenters(:, 2) >= (c-1)*sectorSz(2)+1 & rfCenters(:, 2) < c*sectorSz(2)+1;
          % Assign label to all cells in the sector
          obj.inputSectorLabels(cond) = sectorCounter;
          % Record number of cells (features) in each sector
          obj.numFeats(sectorCounter) = sum(cond);
          % Update sector label
          sectorCounter = sectorCounter + 1;
        end
      end

      % Create grid of RF positions
      rfLinInds = 1:prod(obj.dims);
      [g_r, g_c] = ind2sub(obj.dims, rfLinInds);
      obj.rfCenters2D = [g_r; g_c]';
    end

    function initializeSectorObjects(obj)
      numSectors = prod(obj.params.num_sectors);

      % Create array of fuzzy ART sector cells
      obj.sectorUnits = FuzzyART.empty(numSectors, 0);
      for s = 1:numSectors
        obj.sectorUnits(s) = FuzzyART(numFeats=obj.numFeats(s), params=obj.params.art);
      end
    end

    function setParams(obj, params)
      obj.params = params;

      for i = 1:numel(obj.sectorUnits)
        obj.sectorUnits(i).setParams(params.art);
      end
    end

    function train(obj, inputs)
      % Train each sector on activation that arises therein
      % Uneven number of input features (e.g. from previous SectorLayer) -> flatten
      if iscell(inputs)
        inputs = cell2mat(inputs')';
      end

      % Apply input transformation
      inputs = obj.transformInput(inputs);

      for s = 1:numel(obj.sectorUnits)
        obj.sectorUnits(s).train(inputs(obj.inputSectorLabels == s, :));

        if obj.params.art.verbose
          fprintf('Sector %d: Num committed cells = %d\n', s, obj.sectorUnits(s).getNumCommitted());
        end
      end

      numCommittedPerSector = arrayfun(@(cell) cell.C, obj.sectorUnits);
      disp('*Number of committed units in sectors: ');
      disp(numCommittedPerSector);
      disp('*Mean number across layer: ');
      disp(mean(numCommittedPerSector));

      % Plot the learned weights? Only really useful for MSTd
      if obj.params.plot.weights
        obj.plotWtsGrid();
      end
    end

    function act = predict(obj, inputs)
      numSectors = numel(obj.sectorUnits);

      act = cell(numSectors, 1);
      wts = cell(numSectors, 1);

      % Apply input transformation
      inputs = obj.transformInput(inputs);

      for s = 1:numSectors
        % Filter out only inputs to the current sector
        sectorInput = inputs(obj.inputSectorLabels == s, :);
        % For each sector and input, get the activity distribution
        act{s} = obj.sectorUnits(s).predict(sectorInput);
        % Look up the weights corresponding each sector
        if strcmpi(obj.params.art.predict_fun, 'wta')
          % WTA: Take winner's bottom-up wts
          wts{s} = obj.sectorUnits(s).getWts(argmax(act{s}, 2));
        else
          % Distributed: Do weighted sum of coding cell wts
          currWts = obj.sectorUnits(s).getWts();
          currWts = reshape(currWts, [size(currWts, 1), 1, size(currWts, 2)]);
          wts{s} = sum(shiftdim(act{s}, -1) .* currWts, 3);
        end
      end

      obj.act = act;
      obj.wts = wts;
    end

    function exportStruct = getExportData(obj)
      exportStruct = struct();
    end

    function [sectorDirs, sectorDirsX, sectorDirsY] = decode(obj, wts, numSamples)
      numSectors = numel(wts);
      sectorDirs = zeros(numSectors, numSamples);
      sectorDirsX = zeros(numSectors, numSamples);
      sectorDirsY = zeros(numSectors, numSamples);
      for s = 1:numSectors
        % Normalize weights in each sector
        currWtsNorm = wts{s}(:, 1:numSamples);
        currWtsNorm = currWtsNorm ./ sum(currWtsNorm);
        % Get direction preferences of MT cells factoring into each sector
        currDirPrefs = obj.inputLayer.dirPrefs(obj.inputSectorLabels == s);

        % Population vector (centroid) decoding of dominant direction in the sector
        wt_sum_x = sum(currWtsNorm .* cosd(currDirPrefs)) ./ sum(currWtsNorm + eps);
        wt_sum_y = sum(currWtsNorm .* sind(currDirPrefs)) ./ sum(currWtsNorm + eps);
        est_dirs = atan2d(wt_sum_y, wt_sum_x);

        % Record readout
        sectorDirsX(s, :) = wt_sum_x;
        sectorDirsY(s, :) = wt_sum_y;
        sectorDirs(s, :) = est_dirs;
      end
    end

    function plotWtsGrid(obj)
      % Assuming we are applying this to MSTd-only right now
      numCommitted = obj.sectorUnits(1).getNumCommitted();

      maxCols = 5;
      figure();

      for c = 1:numCommitted
        [vec_dx, vec_dy] = obj.getWtVectors(c);
        
        % Plot vector in each sector
        subplot(ceil(numCommitted/maxCols), maxCols, c);
        quiver(obj.inputLayer.rfCenters2D(:, 2), obj.inputLayer.rfCenters2D(:, 1), vec_dx, vec_dy)
      end

      sgtitle([obj.label, ' (committed cell wts)']);
    end

    function plotWtRow(obj, rowWidth)
      if nargin < 2
        rowWidth = obj.sectorUnits(1).getNumCommitted();
      end

      for c = 1:rowWidth
        [vec_dx, vec_dy] = obj.getWtVectors(c);
        
        % Plot vectors of each template
        nexttile;
        quiver(obj.inputLayer.rfCenters2D(:, 2), obj.inputLayer.rfCenters2D(:, 1), vec_dx, vec_dy)
        
        % Customize appearance
        xlim([0, obj.inputLayer.dims(2)+1])
        ylim([0, obj.inputLayer.dims(1)+1])
        xticks([]);
        yticks([]);
      end
    end

    function [vec_dx, vec_dy] = getWtVectors(obj, c)
      % Container for all wts corresponding to committed cell c in each sector
      wts = cell(numel(obj.sectorUnits), 1);

      % Get the learned committed wts in each sector
      for s = 1:numel(obj.sectorUnits)
        wts{s} = obj.sectorUnits(s).getWts(c);
      end

      % Get the weights of the first (and assumed only) sector in current layer
      if isa(obj.inputLayer, 'ARTSectorLayer')
        % Propogate weights back a layer
        wts = obj.avgFeedbackWts(wts{1});
      end

      % Get directions within each sector
      [~, vec_dx, vec_dy] = obj.inputLayer.decode(wts, 1);
    end


    function predictInput(obj, numPlotSamples)
      arguments
        obj
        numPlotSamples double = 20
      end

      % Predict each input
      % Get the weights of the first (and assumed only) sector in current layer
      if isa(obj.inputLayer, 'ARTSectorLayer')
        % Propogate weights back a layer
        % NOTE: obj.wts set by obj.predict, which bakes in activation into wts.
        wts = obj.avgFeedbackWts(obj.wts{1});

        numSamples = size(obj.act{1}, 1);
        N = min(numSamples, numPlotSamples);

        % Get directions within each sector
        [~, sectorDirsX, sectorDirsY] = obj.inputLayer.decode(wts, N);

        % Plot vector in each sector
        for i = 1:N
          nexttile;
          quiver(obj.inputLayer.rfCenters2D(:, 2), obj.inputLayer.rfCenters2D(:, 1), sectorDirsX(:, i), sectorDirsY(:, i))

          % Customize appearance
          xlim([0, obj.inputLayer.dims(2)+1])
          ylim([0, obj.inputLayer.dims(1)+1])
          xticks([]);
          yticks([]);
        end
      else
        % Get directions within each sector
        [~, sectorDirsX, sectorDirsY] = obj.decode(obj.wts, inputs);

        % Plot vector in each sector
        for i = 1:min(inputs, maxRows*maxCols)
          nexttile;
          quiver(obj.rfCenters2D(:, 2), obj.rfCenters2D(:, 1), sectorDirsX(:, i), sectorDirsY(:, i))
        end
      end
    end

    function avgWts = avgFeedbackWts(obj, wts)
      % To visualize the templates, we want to derive one set of weights per sector in the previous layer.
      % We do wted average the weights in the previous layer
      numInputLayerSectors = prod(obj.inputLayer.dims);
      avgWts = cell(1, numInputLayerSectors);
      for s = 1:numInputLayerSectors
        sectorWts = obj.inputLayer.sectorUnits(s).getWts();
        sectorWts = reshape(sectorWts, [size(sectorWts, 1), 1, size(sectorWts, 2)]);
        fbWts = wts(obj.fbSectorLabels == s, :)';
        avgWts{s} = sum(sectorWts .* shiftdim(fbWts, -1), 3);
      end
    end

    function inputs = transformInput(obj, inputs)
      %transformInput Apply transformation to each sector within each sample of inputs
      if ~any(structfun(@(x) x.do, obj.params.input_transform))
        return;
      end

      numSectors = numel(obj.sectorUnits);

      for s = 1:numSectors
        currSectInput = inputs(obj.inputSectorLabels == s, :);

        if obj.params.input_transform.sigmoid.do
          currSectInput = currSectInput ./ (eps + sum(currSectInput));
          mid = obj.params.input_transform.sigmoid.mid;
          currSectInput = currSectInput.^2 ./ (currSectInput.^2 + mid^2 + eps);
        end

        inputs(obj.inputSectorLabels == s, :) = currSectInput;
      end
    end
  end
end