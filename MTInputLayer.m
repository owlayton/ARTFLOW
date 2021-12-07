classdef MTInputLayer < DynamicLayer
  %MTInputLayer

  properties
    dx
    dims
    rfCenters2D  % Stored in (row, col) convention
    rfInds % num_inds x num_cells
    dirPrefs
    spdSigmas
    spdOffsets
    spdPrefs
    netInput
  end

  methods
    function obj = MTInputLayer(nameValueArgs)
      %MTInputLayer Construct an instance of this class
      arguments
        nameValueArgs.params struct
        nameValueArgs.odeConfig struct
        nameValueArgs.dims (1, 2) double
        nameValueArgs.initialize logical = true
      end

      % Create instance vars for layer params and ODE config in a consistent way across layers
      obj = obj@DynamicLayer(params=nameValueArgs.params, odeConfig=nameValueArgs.odeConfig);

      obj.dims = nameValueArgs.dims;

      % Initialize units in layer
      if nameValueArgs.initialize
        obj.initialize()
      end
    end

    function initialize(obj)
      % Get the MT RF radius
      radius = obj.params.radius;

      % Generate RF locations
      % We want them to lie within a RF radius of the border
      rf_cent_r = randi(obj.dims(1)-2*radius, obj.params.num_cells, 1) + radius;
      rf_cent_c = randi(obj.dims(2)-2*radius, obj.params.num_cells, 1) + radius;
      obj.rfCenters2D = [rf_cent_r, rf_cent_c];

      % Record (x,y) indices inside each RF as linear indices for simulation runtime efficiency
      [g_c, g_r] = meshgrid(1:obj.dims(2), 1:obj.dims(1));
      for i = 1:obj.params.num_cells
        % Collect indices within a radius of the current cell's RF center
        % L1 distance version
        % dists = abs(g_r - rf_cent_r(i)) + abs(g_c - rf_cent_c(i));
        % L2 distance version
        dists = sqrt((g_r - rf_cent_r(i)).^2 + (g_c - rf_cent_c(i)).^2);

        % Note: this is strictly less than radius, meaning there will be radius-1 cells to either side of center
        % bilaterally
        rows = g_r(dists < radius);
        cols = g_c(dists < radius);

        % Once we know the number of cells in one RF, we can initialize the array to store them for all cells
        if i == 1
          obj.rfInds = zeros(numel(rows), obj.params.num_cells);
        end

        obj.rfInds(:, i) = sub2ind(obj.dims, rows, cols);
      end

      % Establish von Mises direction tuning preferred directions
      % Uniform sampling of preferred speeds in range [-180 deg, +180 deg]
      obj.dirPrefs = 360*rand(obj.params.num_cells, 1) - 180;

      % Initialize speed preferences
      obj.initializeSpds();
    end

    function initializeSpds(obj)
      % Establish log Gaussian speed tuning parameters
      % 1) Tuning width sigma: Sample from approximate distribution from Nover et al. (2005) Fig 4a
      sig_mean = obj.params.speed.sigma.mean;
      sig_std = obj.params.speed.sigma.std;

      obj.spdSigmas = sig_std*randn(obj.params.num_cells, 1) + sig_mean;
      while any(obj.spdSigmas <= 0)
        obj.spdSigmas(obj.spdSigmas <= 0) = sig_std*randn(nnz(obj.spdSigmas <= 0), 1) + sig_mean;
      end

      % 2) speed offset parameter used to ensure positivity of lognormal
      % Assume that it follows an expotential distribution
      offset_lambda = obj.params.speed.offset.lambda;
      obj.spdOffsets = exprnd(offset_lambda, obj.params.num_cells, 1);

      % 3 Generate preferred speeds
      num_bins = obj.params.speed.preferred_speed.num_bins;
      first_bin_width = obj.params.speed.preferred_speed.first_bin_width;
      min_spd = obj.params.speed.preferred_speed.min;
      max_spd = obj.params.speed.preferred_speed.max;

      % Make octave spaced bins
      if strcmpi('octave', obj.params.speed.preferred_speed.method)
        bin_edges = zeros(num_bins+1, 1);
        bin_edges(1) = min_spd;
        for i = 2:num_bins
          bin_edges(i) = first_bin_width^(i-1) + bin_edges(i-1);
        end
        bin_edges(end) = max_spd;
      elseif strcmpi('logspace', obj.params.speed.preferred_speed.method)
        % logspace method
        bin_edges = logspace(log10(min_spd), log10(max_spd), num_bins+1);
      end

      % Sample speed preferences uniformly in each speed bin
      obj.spdPrefs = zeros(obj.params.num_cells, 1);
      cursor = 0;
      for i = 1:num_bins
        % If the number of speed bins doesn't cleanly divide into number of cells, fill the last bin with more samples
        if i < num_bins
          numSampled = floor(obj.params.num_cells / num_bins);
        else
          numSampled = obj.params.num_cells - (i-1)*floor(obj.params.num_cells / num_bins);
        end

        obj.spdPrefs(cursor+1:cursor+numSampled) = (bin_edges(i+1) - bin_edges(i))*rand(numSampled, 1) + bin_edges(i);
        cursor = cursor + numSampled;
      end
    end

    function netIn(obj, inputs, simParams)
      % Only compute the net input to MT based on optic flow inputs on the first time step of each frame (doesn't
      % change)
      if simParams.t_step ~= 1
        return
      end

      % Resolve net input
      for c = 1:obj.params.num_cells
        % Get indices inside current cell's RF
        inds = obj.rfInds(:, c);

        % Evaluate direction match between input and each MT cell
        dirMatch = vonMises(inputs.dirs(inds), obj.params.direction.sigma.value, obj.dirPrefs(c));

        % Evaluate speed match between input and MT cells
        spdMatch = logNormal(inputs.spds(inds), obj.spdSigmas(c), obj.spdOffsets(c), obj.spdPrefs(c));

        % netInput: product of speed and direction match averaged over number of inputs, suppressing 0 speeds
        obj.netInput(c) = mean((inputs.spds(inds) > eps) .* dirMatch .* spdMatch);
      end

      if obj.params.plot.net_input
        obj.plotVectors(obj.netInput);
      end
    end

    function evaluate(obj, simParams)
      % Only compute the net input to MT based on optic flow inputs on the first time step of each frame (doesn't
      % change)
      if simParams.t_step ~= 1
        return
      end

      obj.dx = -obj.params.ode.decay_rate*obj.act + (obj.params.ode.upper_bound - obj.act).*obj.netInput;
    end

    function updateActivation(obj, simParams)
      obj.act = obj.act + (1/simParams.num_time_steps)*obj.dx;
    end

    function act = getActivation(obj)
      act = obj.act;
    end

    function exportStruct = getExportData(obj)
      exportStruct = struct();
      exportStruct.label = obj.label;
      exportStruct.rfCenters2D = obj.rfCenters2D;
      exportStruct.rfInds = obj.rfInds;
      exportStruct.dirPrefs = obj.dirPrefs;
      exportStruct.spdPrefs = obj.spdPrefs;
      exportStruct.act = obj.act;
    end

    function plotVectors(obj, act)
      %plotVectors plots the layer activation as a quiver vector field plot wherein arrow length indicates strength of
      %activation and direction indicates the unit's direction preference.
      % Parameters:
      %
      % act: the activation to be plotted. Usually netInput or activation.
      figure(1);
      clf;
      % Create arrays for x and y vector components
      vec_x = zeros(obj.dims);
      vec_y = zeros(obj.dims);

      % Map 2D RF coordinates (row cols) into linear indices in the input space
      rfLinearInds = sub2ind(obj.dims, obj.rfCenters2D(:, 1), obj.rfCenters2D(:, 2));

      % Work out horizontal (cos) and vertical (sin) vector components
      vec_x(rfLinearInds) = cosd(obj.dirPrefs) .* act;
      vec_y(rfLinearInds) = sind(obj.dirPrefs) .* act;

      % Create grid consistent with RF placement scheme
      [g_c, g_r] = meshgrid(1:obj.dims(2), 1:obj.dims(1));
      % Draw vectors
      quiver(g_c, g_r, vec_x, vec_y, 30)

      % Set plotting bounds
      pad = 30;
      xlim([1-pad, obj.dims(2)+pad]);
      ylim([1-pad, obj.dims(1)+pad]);
    end

    function plotVectorsGrid(obj, gridDims, inputSectorLabels, acts, numPlotted)
      arguments
        obj
        gridDims (1, :) double
        inputSectorLabels (:, 1) double
        acts double = obj.act
        numPlotted double = 20
      end
      % Set the activation so that the object methods can access it
      obj.act = acts;
      % Decode the motion in each sector
      [vec_x, vec_y, vec_dx, vec_dy] = obj.getGridVectors(gridDims, inputSectorLabels);
      % Plot it
      for i = 1:min(numPlotted, size(acts, 2))
        nexttile;
        quiver(vec_y, vec_x, vec_dx(i, :), vec_dy(i, :));

        % Customize appearance
        xlim([0, gridDims(2)+1])
        ylim([0, gridDims(1)+1])
        xticks([]);
        yticks([]);
      end
    end

    function [dx, dy] = decodeSector(obj, actSectorMask)
      % Activation in each sector
      currActNorm = obj.act(actSectorMask, :);
      % Get direction preferences of MT cells factoring into each sector
      currDirPrefs = obj.dirPrefs(actSectorMask);

      % Population vector (centroid) decoding of dominant direction in the sector
      dx = sum(currActNorm .* cosd(currDirPrefs)) ./ sum(currActNorm + eps);
      dy = sum(currActNorm .* sind(currDirPrefs)) ./ sum(currActNorm + eps);
    end

    function [vec_x, vec_y, vec_dx, vec_dy] = getGridVectors(obj, gridDims, inputSectorLabels)
      numSectors = prod(gridDims);
      numSamples = size(obj.act, 2);
      
      [vec_x, vec_y] = ind2sub(gridDims', 1:prod(gridDims));

      vec_dx = zeros(numSamples, numSectors);
      vec_dy = zeros(numSamples, numSectors);

      for s = 1:numSectors
        % Select cells inside the current grid sector & get population decoded direction
        [curr_dx, curr_dy] = obj.decodeSector(inputSectorLabels == s);
        vec_dx(:,s) = curr_dx;
        vec_dy(:,s) = curr_dy;
      end
    end
  end
end

function wt = vonMises(inputDir, bw, dirPref)
  wt = exp(bw .* (cosd(inputDir - dirPref) - 1));
end

function wt = logNormal(inputSpd, sigma, offset, spdPref)
  numer = log((inputSpd + offset) ./ (spdPref + offset)).^2;
  denom = 2*sigma.^2;
  wt = exp(-numer ./ denom);
end