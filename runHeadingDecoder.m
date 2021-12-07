function [preds, trainedNet] = runHeadingDecoder(runModes, namedArgs)
  arguments
    runModes (1, :) string % "train" and/or "test"
    namedArgs.decoder_config struct = readSettings('decoder.json');
    namedArgs.io_config struct = readSettings();
  end
  
  % Make Heading Decoder Network
  net = HDecoderNet(decode_params=namedArgs.decoder_config, io_config=namedArgs.io_config);

  % Train the decoder (if needed)
  if inStr('train', runModes)
    trainedNet = net.train();
  else
    trainedNet = [];
  end

  % Predict either new samples or validate training samples
  if inStr('test', runModes)
    preds = net.predict(trainedNet);
  else
    preds = [];
  end
end