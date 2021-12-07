function json = readSettings(varargin)
  % Reads in from file, parses, and returns the JSON-formatted config file
  % (e.g. export and stimulus paths).
  %
  % You can run this without arguments if you use the default io.json in the config folder
  
  args = parseInputs(varargin);

  % try to open the settings.json file
  [file, errorMsg] = fopen(fullfile(args.path, args.filename));
  
  % throw an error if we can't find it
  if ~isempty(errorMsg)
    error('Cannot find your config JSON file!');
  end
  
  % convert file import to string
  fileStr = char(fread(file, inf)');
  % close file handle
  fclose(file);
  
  % parse JSON settings into a struct
  json = jsondecode(fileStr);
  
  if ~isempty(args.field)
    if isfield(json, args.field)
      json = json.(args.field);
    else
      fprintf('WARNING: Requested substruct name %s but it does not exist in config.', args.(subStructName));
    end
  end
end

function argStruct = parseInputs(argCell)
  %parseInputs handle parameter overrides and defaults
  % Handle parsing args and setting param defaults...
  args = inputParser;
  addOptional(args, 'filename', 'io.json', @(x) ischar(x) || isstring(x));
  addParameter(args, 'path', 'configs', @(x) ischar(x) || isstring(x));
  addParameter(args, 'field', '', @(x) ischar(x) || isstring(x));
  parse(args, argCell{:});
  
  % Get the struct containing the results of the param parsing
  argStruct = args.Results;
end