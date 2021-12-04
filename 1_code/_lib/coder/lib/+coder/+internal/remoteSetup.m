function [varargout] = remoteSetup(h, p)
    %#codegen

%   Copyright 2019 The MathWorks, Inc.

    persistent host port
    
    if isempty(host)
        host = 'Error';
        port = int32(-1);
    end
    
    if nargin == 0
        % calling remoteSetup() with no inputs returns the 
        % currently set host and port
        varargout{1} = host;
        varargout{2} = port;
    else
        coder.internal.errorIf(~isnumeric(p), 'Coder:builtins:Explicit', 'The port number provided must be numeric');
        host = h;
        port = int32(p);
    end
end
