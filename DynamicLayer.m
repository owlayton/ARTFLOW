classdef (Abstract) DynamicLayer < Layer
  %LAYER Summary of this class goes here
  %   Detailed explanation goes here
  properties
    odeConfig
  end
  
  methods (Abstract)
    netIn(obj, inputs, simParams)
    evaluate(obj, simParams)
    updateActivation(obj)
    getExportData(obj)
  end
  
  methods
    function obj = DynamicLayer(nameValueArgs)
      arguments
        nameValueArgs.params struct
        nameValueArgs.odeConfig struct
      end
      
      obj = obj@Layer(params=nameValueArgs.params);
      obj.odeConfig = nameValueArgs.odeConfig;
    end
    
    function simulate(obj, inputs, simParams)
      %simulate Simulate the current layer of the neural system at the current timestep
      arguments
        obj
        inputs struct
        simParams struct
      end
      
      % Resolve layer inputs
      obj.netIn(inputs, simParams);
      % Evaluate the layer ODE
      obj.evaluate(simParams);
      % Evolve the activation one time step
      obj.updateActivation(simParams);
    end
    
    function clearActivation(obj)
      % Initialize netInput
      obj.netInput = zeros(obj.params.num_cells, 1);
      
      % Initialize the activation
      obj.act = zeros(obj.params.num_cells, 1);
    end
  end
end

