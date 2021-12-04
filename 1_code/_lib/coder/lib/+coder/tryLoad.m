%

%   Copyright 2019 The MathWorks, Inc.


function out = tryLoad(matfile, varargin)
    %fail silently and return nothing

    try
        varStruct = load(matfile);
        out = varStruct.(varargin{:});
    catch
        out = [];
    end


end
