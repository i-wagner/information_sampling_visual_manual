function p = isAllowedSparseClass(s)
%MATLAB Code Generation Private Function

%   Copyright 2017-2018 The MathWorks, Inc.
%#codegen
coder.inline('always');
p = islogical(s) || isa(s,'double') || ischar(s);
