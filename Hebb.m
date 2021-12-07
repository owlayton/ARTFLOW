classdef Hebb < handle
  %HEBB Unit that learns via Hebbian learning
  
  properties
    params
    numFeats
    wts
    n_iter
  end
  
  methods
    function obj = Hebb(nameValueArgs)
      %HEBB Construct an instance of this class
      arguments
        nameValueArgs.params struct
        nameValueArgs.numFeats double
      end

      obj.params = nameValueArgs.params;
      obj.numFeats = nameValueArgs.numFeats;

      % Initialize coding layer wts
      obj.wts = randn(obj.numFeats, obj.params.num_cells);

      % Initialize number of training iterations until convergence
      obj.n_iter = 0;
    end

    function wts = getWts(obj)
      wts = obj.wts;
    end
    
    function train(obj, data)
      % Number of data samples: N
      % Number of features: M
      [M, N] = size(data);

      % Error check numFeats
      if M ~= obj.numFeats
        error('Mismatch between initialized num_features (%d) and shape of data (%d)', M, obj.numFeats);
      end

      % Cycle thru data until wts stop changing or we reach max epochs
      prevWts = ones('like', obj.wts);
      while norm(obj.wts - prevWts, 'fro') > obj.params.threshold
        % Update number of iter taken until convergence
        obj.n_iter = obj.n_iter + 1;

        dWts = zeros('like', obj.wts);
        
        % Loop thru samples, accumulate wt change
        for i = 1:N
          currSamp = data(:, i);
          act = obj.wts' * currSamp;
          dWts = dWts + obj.params.lr * (act*currSamp' - tril(act * act')*obj.wts')';
        end

        % Update prev wts
        prevWts = obj.wts;

        % Update wts
        obj.wts = obj.wts + (obj.params.lr / obj.n_iter)*dWts;
        % Normalize wts
        obj.wts = obj.wts ./ (eps + sqrt(sum(obj.wts.^2, 1)));

        if obj.params.verbose
          fprintf("Iter %d: Norm = %.2f\n", obj.n_iter, norm(obj.wts - prevWts, 'fro'));
        end
      end
    end

    function preds = predict(obj, data)
      % Number of data samples: N
      % Number of features: M
      [M, N] = size(data);

      % Error check numFeats
      if M ~= obj.numFeats
        error('Mismatch between initialized num_features (%d) and shape of data (%d)', M, obj.numFeats);
      end

      preds = (obj.wts' * data)';

      if strcmpi('logistic', obj.params.act_fun)
        preds = 1 ./ (1 + exp(-preds));
      end
    end

    function numCells = getNumCommitted(obj)
      % Compatability with fuzzy ART
      numCells = obj.params.num_cells;
    end
  end
end

