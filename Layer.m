classdef (Abstract) Layer < handle
  %LAYER Summary of this class goes here
  %   Detailed explanation goes here
  properties
    params
    label  % string label used for identifying layer data (exporting, etc).
    act
  end
  
  
  methods
    function obj = Layer(nameValueArgs)
      arguments
        nameValueArgs.params struct
      end
      
      obj.params = nameValueArgs.params;
      obj.label = obj.params.label;
    end

    function setParams(obj, params)
      obj.params = params;
    end
  end
end
