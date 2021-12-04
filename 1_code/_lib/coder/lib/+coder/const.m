function varargout = const(varargin)
%CODER.CONST evaluates an expression or function call at compile time.
%
%   CODER.CONST(EXPR) evaluates expression EXPR. This can handle simple
%   function calls, e.g. A = coder.const(fcn(10)).
%
%   [A1,...,An] = CODER.CONST(@FCN, ARG1, ..., ARGn) calls function @FCN with
%   multiple output arguments.
%

%   Copyright 2013-2019 The MathWorks, Inc.

if nargin == 0
    error(message('Coder:common:NotEnoughInputs'));
end
if isa(varargin{1}, 'function_handle')
    % [A] = CODER.CONST(@FCN)
    if nargin == 1 && nargout <= 1
        varargout{1} = varargin{1};
        return
    end

    % Otherwise apply a function call
    varargout = cell(1,nargout);
    [varargout{:}] = feval(varargin{1}, varargin{2:end});
else
    if nargin > 1
        error(message('Coder:common:TooManyInputs'));
    end
    % A = CODER.CONST(<EXPR>);
    varargout{:} = feval(@()(varargin{1}));
end
