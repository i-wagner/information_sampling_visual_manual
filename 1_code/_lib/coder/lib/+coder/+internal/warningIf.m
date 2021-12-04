function warningIf(cond, msgID, varargin)
%MATLAB Code Generation Private Function

%   Copyright 2011-2019 The MathWorks, Inc.

%MATLAB execution
if cond
    coder.internal.warning(msgID, varargin{:});
end
