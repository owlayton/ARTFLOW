function simulateModel
  %simulateModel Run this function to train ARTFLOW and predict heading on the 3D dot cloud (T) test set.
  % 3D dot cloud (T) included in data/DotCloud3DT
  % Note: The MT preprocessing can take 5+ mins depending on your machine and whether you have parallelization turned
  % on. The decoding (gradient descent) also may take a number of minutes to complete.

  % Close any open figures
  close all hidden;

  % Make ModelSimulation
  sim = ModelSimulation();

  % Run the simulation
  sim.run();
end