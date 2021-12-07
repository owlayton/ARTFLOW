classdef HebbSectorLayer < ARTSectorLayer
  %HebbSectorLayer - cells learn in each sector according to Hebb's Rule
  methods
    function initializeSectorObjects(obj)
      numSectors = prod(obj.params.num_sectors);

      % Create array of fuzzy ART sector cells
      obj.sectorUnits = Hebb.empty(numSectors, 0);
      for s = 1:numSectors
        obj.sectorUnits(s) = Hebb(numFeats=obj.numFeats(s), params=obj.params.hebb);
      end
    end

    function train(obj, inputs)
      % Train each sector on activation that arises therein
      if iscell(inputs)
        inputs = cell2mat(inputs')';
      end

      % Apply input transformation
      inputs = obj.transformInput(inputs);

      for s = 1:numel(obj.sectorUnits)
        obj.sectorUnits(s).train(inputs(obj.inputSectorLabels == s, :));
      end
    end

    function act = predict(obj, inputs)
      numSectors = numel(obj.sectorUnits);

      % Act shape: num_sectors x C x N
      act = cell(numSectors, 1);
      % Wt shape: num_sectors x C x N
      wts = cell(numSectors, 1);

      % Apply input transformation
      inputs = obj.transformInput(inputs);

      for s = 1:numSectors
        % Filter out only inputs to the current sector
        sectorInput = inputs(obj.inputSectorLabels == s, :);
        % For each sector and input, get the activity distribution
        act{s} = obj.sectorUnits(s).predict(sectorInput);

        % Distributed: Do weighted sum of coding cell wts
        currWts = obj.sectorUnits(s).getWts();
        currWts = reshape(currWts, [size(currWts, 1), 1, size(currWts, 2)]);
        wts{s} = sum(shiftdim(act{s}, -1) .* currWts, 3);
      end
      obj.act = act;
      obj.wts = wts;
    end
  end
end