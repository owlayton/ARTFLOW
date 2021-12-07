function actStruct = transposeActStruct(results)
  %resultsStruct2LayerActs "Transposes" the results struct: Fields go from sample# to layerName
  actStruct = struct();
  
  stimuliNames = fieldnames(results);
  layerNames = fieldnames(results.(stimuliNames{1}));
  
  for l = 1:numel(layerNames)
    for s = 1:numel(stimuliNames)
      actStruct.(layerNames{l}).(stimuliNames{s}) = results.(stimuliNames{s}).(layerNames{l});
    end
  end
end

