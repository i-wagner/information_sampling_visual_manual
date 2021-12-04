function p = isCharOrScalarString(s)
%MATLAB Code Generation Private Function

%   Copyright 2016-2019 The MathWorks, Inc.

coder.internal.allowEnumInputs;
coder.internal.allowHalfInputs;
coder.internal.prefer_const(s);
p = ischar(s) || (isstring(s) && isscalar(s));
