function mergedStruct = mergeDistResults(resultsCell)
  %MERGEDISTRESULTS Builds merged struct array from distributed results cell array
  mergedStruct = struct();
  
  for w = 1:numel(resultsCell)
    currResultsStruct = resultsCell{w};
    resultsNames = fieldnames(currResultsStruct);
    
    for r = 1:numel(resultsNames)
      layerNames = fieldnames(currResultsStruct.(resultsNames{r}));
      for a = 1:numel(layerNames)
        mergedStruct.(resultsNames{r}).(layerNames{a}) = currResultsStruct.(resultsNames{r}).(layerNames{a});
      end
    end
  end
end

