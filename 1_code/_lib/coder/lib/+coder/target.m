function y = target(varargin)
%CODER.TARGET Determine the current code-generation target
%
%   CODER.TARGET('TARGET') determines if the specified TARGET is the
%   current code generation target. The following TARGET values may be
%   specified:
%       MATLAB      True if running in MATLAB (not generating code).
%       MEX         True if generating a MEX function.
%       Sfun        True if simulating a Simulink model.
%       Rtw         True if generating a LIB, DLL or EXE target.
%       HDL         True if generating an HDL target.
%       Custom      True if generating a custom target.
%
%   Example:
%       if coder.target('MATLAB')
%           % code for MATLAB evaluation
%       else
%           % code for code generation
%       end
%
%   See also coder.ceval.

%   Copyright 2006-2019 The MathWorks, Inc.

if nargin == 0
    % Backward compatibility
    y = '';
else
    target = varargin{1};
    y = isempty(target) || ...
        ((ischar(target) || (isstring(target) && isscalar(target))) ...
            && strcmpi(target, 'MATLAB'));
end
