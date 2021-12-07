function inputStruct = getCurrInput(sceneStruct, obsStruct, frameNum, samplePath)
  arguments
    sceneStruct struct
    obsStruct struct
    frameNum double
    samplePath char = ''
  end
  %GETCURRINPUT Returns a struct containing the current input. For optic flow, this is x, y, dx, dy
  frameLabel = getFrameLabel(frameNum, sceneStruct.fileNumDigits);

  % Process input based on data format (airsim vs analytic)
  switch sceneStruct.fileFormat
    case 'analytic'
      inputStruct = getAnalyticInput(frameLabel, sceneStruct, obsStruct);
    case 'airsim'
      inputStruct = getAirSimInput(frameLabel, samplePath, obsStruct.fov, sceneStruct.fps);
  end
end

function inputStruct = getAnalyticInput(frameLabel, sceneStruct, obsStruct)
  % MT input layer: directions (deg) and speed (deg/sec) for each pixel of current frame, represented as linear indices
  dims = sceneStruct.dims;
  flow_x = sceneStruct.x.(frameLabel);
  flow_y = sceneStruct.y.(frameLabel);
  flow_dx = sceneStruct.dx.(frameLabel);
  flow_dy = sceneStruct.dy.(frameLabel);
  
  % Compute linear indices of flow vectors: sub2ind(dims, rows, cols)
  flow_inds = sub2ind(dims, flow_y, flow_x);
  
  % 1) direction of each flow vector (deg: -180->+180)
  dirs = computeDirection(flow_dx, flow_dy);
  inputStruct.dirs = zeros(prod(dims(1:2)), 1);
  inputStruct.dirs(flow_inds) = dirs;
  
  % 2) speed of each flow vector (px/frame)
  fov = obsStruct.fov.(frameLabel);
  fps = sceneStruct.fps;
  spds = computeSpeed(flow_dx, flow_dy, fov, dims, fps);
  inputStruct.spds = zeros(prod(dims(1:2)), 1);
  inputStruct.spds(flow_inds) = spds;
end

function inputStruct = getAirSimInput(frameLabel, samplePath, fov, fps)
  % Load the current frame file from disk
  frameFilePath = fullfile(samplePath, 'OpticFlow', [frameLabel, '.mat']);
  load(frameFilePath, 'dx', 'dy', 'dims');

  % Fix dims datatype
  dims = double(dims);
  
  % Transpose and flatten the 2D flow
  dx = dx';
  dy = dy';
  dx = dx(:);
  dy = dy(:);

  % Compute directions
  inputStruct.dirs = computeDirection(dx, dy);
  inputStruct.spds = computeSpeed(dx, dy, fov, dims, fps);
end

function dirs = computeDirection(flow_dx, flow_dy)
  dirs = atan2d(flow_dy, flow_dx);
end

function spds = computeSpeed(flow_dx, flow_dy, fov, dims, fps)
  % Compute speed of each flow vector (px/frame)
  spds = sqrt(flow_dx.^2 + flow_dy.^2);
  % Add speed computation of each vector (deg/sec)
  pixHypot = sqrt(dims(1).^2 + dims(2).^2);
  degHypot = sqrt(2*(fov/2).^2);
  % deg/sec = pix/frame * frame/sec * deg/pix
  spds = spds * (fps * (degHypot/pixHypot));
end
