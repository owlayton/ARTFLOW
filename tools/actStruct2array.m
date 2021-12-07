function data = actStruct2array(resultsStruct, layerName)
  %actStruct2array Takes activation struct array and converts it to a double matrix
  %or cell array for uneven spaced features
  % data: 2D M x N data array from MT Input activations
  arguments
    resultsStruct struct
    layerName char = 'none' 
  end
  resultsNames = fieldnames(resultsStruct);
  
  if strcmpi(layerName, 'none')
    firstAct = resultsStruct.(resultsNames{1}).act;
    numCells = numel(resultsStruct.(resultsNames{1}).act);
  else
    firstAct = resultsStruct.(resultsNames{1}).(layerName).act;
    numCells = numel(resultsStruct.(resultsNames{1}).(layerName).act);
  end
  
  if iscell(firstAct)
    % Uneven features
    data = cell(numCells, numel(resultsNames));
  else
    data = zeros(numCells, numel(resultsNames));
  end

  for i = 1:numel(resultsNames)
    if strcmpi(layerName, 'none')
      currAct = resultsStruct.(resultsNames{i}).act;
    else
      currAct = resultsStruct.(resultsNames{i}).(layerName).act;
    end

    data(:, i) = currAct;
  end
end

