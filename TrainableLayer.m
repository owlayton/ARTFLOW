classdef (Abstract) TrainableLayer < Layer
  %LAYER Summary of this class goes here
  %   Detailed explanation goes here
  properties
    wts
  end

  methods (Abstract)
    train(obj, inputs)
    predict(obj, inputs)
    getExportData(obj)
  end
  
  methods
    function obj = TrainableLayer(nameValueArgs)
      arguments
        nameValueArgs.params struct
      end
      
      obj = obj@Layer(params=nameValueArgs.params);
    end
  end
end

