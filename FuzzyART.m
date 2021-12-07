classdef FuzzyART < handle
  %FUZZYART Summary of this class goes here
  %   Detailed explanation goes here

  properties
    params
    numFeats  % Number of input features
    C  % int. Num committed neurons
    w_code  % array. size=(2*M, C_max). Input-to-coding-layer adaptive weights.
    num_updates % array. (C_max, 1). Keeps track of number of weight updates experienced by each coding cell
  end

  methods
    function obj = FuzzyART(nameValueArgs)
      %FUZZYART Construct an instance of this class
      arguments
        nameValueArgs.params struct
        nameValueArgs.numFeats double
      end

      obj.params = nameValueArgs.params;
      obj.numFeats = nameValueArgs.numFeats;

      % Initialize coding layer wts
      obj.w_code = ones(2*obj.numFeats, obj.params.C_max);
      % Initialize coding layer wt update tracker
      obj.num_updates = zeros(obj.params.C_max, 1);
    end

    function train(obj, data)
      % Parameters:
      %%%%%%%%%%%%%%%%%%%%
      % data: matrix. size=(#dimensions (M), #samples (N)). Data samples normalized in the range [0,1].

      % Number of data samples: N
      % Number of features: M
      [M, N] = size(data);

      % Error check numFeats
      if M ~= obj.numFeats
        error('Mismatch between initialized num_features (%d) and shape of data (%d)', M, obj.numFeats);
      end

      % Preprocess input
      %
      % Complemenet coding yields size of (2M x N)
      A = obj.complementCode(data);

      % Initialize current number of committed units / "frontier" of committed units
      obj.C = 0;

      for ep = 1:obj.params.num_epochs
        for n = 1:N
          % Get the n-th input
          curr_A = A(:, n);

          % Calculate net input to coding layer
          Tj = obj.choiceByDifference(curr_A, obj.C);

          % Sort possible category matches max->min
          [~, pm_sorted_inds] = sort(Tj, 'descend');

          % Add in next uncommitted node index to list to pick it as a last resort
          pm_sorted_inds = [pm_sorted_inds, obj.C+1];

          % Search for a supra-thresold coding layer match in the committed nodes
          no_resonance = true;
          % Search index among supra-thresold matches
          s = 1;
          % Search for a correct match (resonance) as long as we still have candidates (B.9)
          while no_resonance && s <= numel(pm_sorted_inds)
            % Active code index (1,...,Z): Trace back native coding layer index:
            % curr sorted supra-thres ind -> unsorted supra-thres ind
            active_ind = pm_sorted_inds(s);
            fuzzy_and = sum(abs(min(curr_A, obj.w_code(:, active_ind))));

            matchScore = fuzzy_and/M;

            if obj.params.verbose
              if s == 1
                fprintf('%d. First match score = %.3f\n', n, matchScore);
              end
            end

            % If we have a match...
            if matchScore >= obj.params.p
              % We got it right, stop searching, update weights
              no_resonance = false;

              % fast learning, slow recode
              beta = obj.params.beta;
              if obj.params.do_fast_learning
                % If we've matching a previously commited note, use the lower beta
                if active_ind <= obj.C
                  beta = obj.params.beta_slow_recode;
                else
                  beta = 1;
                end
              end

              % Are we committing a new code?
              if active_ind > obj.C
                obj.C = obj.C + 1;

                % Throw error if we run out of commitable cells
                if obj.C == obj.params.C_max
                  error("Ran out of committable cells!");
                end

                if obj.params.verbose
                  fprintf('Committing new cell on N=%d. Match score was %.3f\n', n, matchScore);
                end
              else
                if obj.params.verbose
                  fprintf('Updating wts on N=%d\n', n);
                end
              end

              % Update the weights for the winner that resonates
              obj.updateWts(curr_A, active_ind, beta);
              % Record that the coding cell had a weight update
              obj.num_updates(active_ind) = obj.num_updates(active_ind) + 1;

              if obj.params.verbose
                fprintf('  Resonance! Coding node %d\n  New wt is ', active_ind);
              end
            end

            s = s + 1;
          end
        end  % END Loop over inputs
      end % END Loop over epochs
    end

    function c_pred = predict(obj, data, method)
      %%Fuzzy ART Predict: Either returns index of winning cell (WTA) or distributed activity across coding layer
      %(CAM rule)
      arguments
        obj
        data (:, :) double
        method char = obj.params.predict_fun
      end

      % Number of data samples: N
      % Number of features: M
      [M, N] = size(data);

      % Preprocess input
      %
      % Complemenet coding yields size of (2M x N)
      A = obj.complementCode(data);

      % Initialize output: class predictions
      %
      c_pred = zeros(N, obj.C);

      for n = 1:N
        % Get the n-th input
        curr_A = A(:, n);

        % Calculate net input to coding layer
        Tj = obj.choiceByDifference(curr_A, obj.C);

        % Make Tj column vector
        Tj = Tj(:);
        % Make sure we only have Tj values for committed cells
        Tj = Tj(1:obj.C);

        % WTA: One-hot coding of winning cell
        if strcmpi(method, 'softmax')
          c_pred(n, :) = softmax(Tj);
        elseif strcmpi(method, 'raw')
          c_pred(n, :) = Tj;
        else
          error('Unknown activation function.');
        end
      end % END Loop over inputs
    end

    function Ac = complementCode(obj, data)
      %%complementCode complement codes data. Size goes from (M, N) -> (2M, N)
      %
      % Parameters:
      %%%%%%%%%%%%%%%%%%%%
      % data: matrix. size = (#dimensions (M), #samples (N)). Data samples normalized in the range [0,1].
      %
      %
      % Returns:
      %%%%%%%%%%%%%%%%%%%%
      % Ac: array. Values in the range [0,1]. size = (2*#dimensions (M), #samples (N)).
      %   1st half along 1st dimension is `data`. 2nd half along dimension is 1-`data`.

      Ac = [data; 1-data];
    end

    function Tj = choiceByDifference(obj, curr_A, C)
      %%choiceByDifference choice-by-difference coding layer net input function
      %
      % Computes 'fuzzy AND' between input pattern and each committed unit's wts.
      % Applies the choice-by-difference function to get netIn for each of the C committed coding units
      %
      % Parameters
      %%%%%%%%%%%%
      % curr_A: matrix. size=(2*M, 1). Current input
      % C: int. Number of committed nodes
      %
      % Returns:
      %%%%%%%%%%%%%%%%%%%%
      % Tj: matrix. size=(1, C). Net input for committed units in the coding layer.
      committed_wts = obj.w_code(:, 1:C);
      fuzzy_and = sum(abs(min(curr_A, committed_wts)));

      Tj = fuzzy_and + (1 - obj.params.alpha)*(obj.numFeats - sum(abs(committed_wts)));
    end

    function updateWts(obj, curr_A, active_ind, beta)
      %%updateWts updates the wts to the active coding unit in `w_code` at index `active_ind`.
      % NOTE: Only updates the active coding unit's wts. The rest remain unchanged.
      % When we have beta=1 (fast learning) the active unit's wts become equal to the 'fuzzy AND' of curr wts and the curr
      % input.
      %
      % Parameters
      %%%%%%%%%%%%
      % beta: double. Wt learning rate
      % curr_A. matrix. size=(2*M, 1). Current input
      % w_code: matrix. size=(2*M, C_max). Input-to-coding-layer adaptive weights.
      % active_ind: int. Index of the currently active coding layer cell.
      %
      % Returns:
      %%%%%%%%%%%%%%%%%%%%
      % w_code: matrix. size=(2*M, C_max). Input-to-coding-layer adaptive weights. Wt updated only for the cell at active_ind.
      fuzzy_and = min(curr_A, obj.w_code(:, active_ind));
      obj.w_code(:, active_ind) = beta*fuzzy_and + (1-beta)*obj.w_code(:, active_ind);
    end

    function wts = getWts(obj, cellInds)
      arguments
        obj
        cellInds double = 1:obj.C
      end
      wts = obj.w_code(1:obj.numFeats, cellInds);
    end

    function numCommitted = getNumCommitted(obj)
      numCommitted = obj.C;
    end

    function setParams(obj, params)
      obj.params = params;
    end
  end
end

