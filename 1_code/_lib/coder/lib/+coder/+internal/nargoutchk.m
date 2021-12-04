function nargoutchk(low,high,n)
%MATLAB Code Generation Library Function

%   Copyright 2012-2019 The MathWorks, Inc.
%#codegen

% Argument checking
coder.internal.prefer_const(low,high,n);
coder.internal.assert(isa(low,'numeric') && isscalar(low) && floor(low) == low, ...
                      'MATLAB:IntVal');
coder.internal.assert(isa(high,'numeric') && isscalar(high) && floor(high) == high, ...
                      'MATLAB:IntVal');
coder.internal.assert(isa(n,'numeric') && isscalar(n) && floor(n) == n, ...
                      'MATLAB:IntVal');

% Ensure nargout is within range
coder.internal.assert(low <= n, ...
                      'MATLAB:nargoutchk:notEnoughOutputs');
coder.internal.assert(high >= n, ...
                      'MATLAB:nargoutchk:tooManyOutputs');
