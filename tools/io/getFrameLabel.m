function [ frameField ] = GetFrameLabel(t, nDigits)
  if nargin < 2
    nDigits = 4;
  end
  frameField = ['Frame' sprintf(['%0', num2str(nDigits), 'd'], t)];
end

