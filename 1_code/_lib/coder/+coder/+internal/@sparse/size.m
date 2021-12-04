function varargout = size(this, dim)
%MATLAB Code Generation Private Method

%   Copyright 2017-2018 The MathWorks, Inc.
%#codegen
coder.inline('always');
if nargin == 2
    % size(a,dim)
    coder.internal.prefer_const(dim);
    nargoutchk(0,1);
    switch dim
      case 1
        varargout{1} = double(this.m);
      case 2
        varargout{1} = double(this.n);
      otherwise
        varargout{1} = double(ONE);
    end
else
    % [...] = size(a)
    if nargout <= 1
        varargout{1} = double([this.m, this.n]);
    else
        varargout{1} = double(this.m);
        varargout{2} = double(this.n);
        for k = 3:nargout
            varargout{k} = double(ONE);
        end
    end
end

%--------------------------------------------------------------------------
