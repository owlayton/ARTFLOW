function cont = inStr(findStr, searchedStr)
  %inStr Shorter way to determine whether a string is contained in another
  %   Usage: inStr(findStr, searchedStr) Find the findStr in one searchedStr
  %   Usage: inStr(findStr, searchedStr) Find the findStr in list
  %   findStr is searched for in searchedStr.
  %   Returns true is found, false if not
  if ~iscell(searchedStr) && ~isstring(searchedStr)
    cont = contains(searchedStr, findStr, 'IgnoreCase', true);
  else
    if isempty(searchedStr)
      cont = false;
    else
      cont = any(cellfun(@(f) contains(f, findStr, 'IgnoreCase', true), searchedStr));
    end
  end
end

