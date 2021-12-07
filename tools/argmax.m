function maxInd = argmax(x, dim)
  %ARGMAX Computes argmax of x
  arguments
    x double
    dim (1,1) double = -1
  end
  
  if dim == -1
    [~, maxInd] = max(x);
  else
    [~, maxInd] = max(x, [], dim);
  end
end

