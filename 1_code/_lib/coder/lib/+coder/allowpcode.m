function allowpcode(varargin)
%CODER.ALLOWPCODE Control code generation from P-files.
%
%   CODER.ALLOWPCODE('plain') allows code generation from the protected
%     MATLAB file (P-file) containing this directive.  No attempt will be
%     made to obfuscate the generated code.
%
%   Example:
%     coder.allowpcode('plain'); % enable code generation from P-file
%
%   This is a code generation function.  It has no effect in MATLAB.
%
%   See also coder.ceval.

%   Copyright 2006-2019 The MathWorks, Inc.

narginchk(1, 2);
valid = false;
for i=1:nargin
    switch varargin{i}
        case 'plain'
            valid = true;
    end
end
if ~valid
    error(message('Coder:builtins:allowpcodeTypeError'));
end
