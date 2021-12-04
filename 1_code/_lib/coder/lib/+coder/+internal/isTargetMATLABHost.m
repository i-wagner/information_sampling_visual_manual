function tf = isTargetMATLABHost()
% tf = isTargetMATLABHost
% Return true if the generated code or this function is running on MATLAB
% host and can take advantage of MATLAB-provided shared libraries.

%   Copyright 2014-2019 The MathWorks, Inc.

%#codegen

% Rapid Acceleration is currently not supported for
% coder.ExternalDependency based codegen. (g983048)
isRapidAccel = coder.target('rtwForRapid');

isMATLABHost = ...
    coder.target('MATLAB') || ...
    coder.target('MEX'   ) || ...
    coder.target('Sfun'  ) || ...
    coder.target('Generic->MATLAB Host Computer') || ...
    coder.target('MATLAB Host');

switch (coder.const(feval('computer', 'arch')))
  case 'win64'
    isIntelPlatform = coder.target('Intel->x86-64 (Windows64)');
  case 'glnxa64'
    isIntelPlatform = coder.target('Intel->x86-64 (Linux 64)');
  case 'maci64'
    isIntelPlatform = coder.target('Intel->x86-64 (Mac OS X)');
  otherwise
    coder.internal.assert(false,'Coder:builtins:Explicit','Internal error: Unsupported platform');
end

tf = (isMATLABHost || isIntelPlatform) && ~isRapidAccel;
end
