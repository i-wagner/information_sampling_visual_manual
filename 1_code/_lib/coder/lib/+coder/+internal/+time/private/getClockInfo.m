function info = getClockInfo(clockname)
%MATLAB Code Generation Private Function

% Returns information needed to call EMLRT/POSIX for the desired clock. Valid
% CLOCKNAME values are: "monotonic", "realtime".

%   Copyright 2019 The MathWorks, Inc.
%#codegen

if clockname == "monotonic"
    posix_clock = "CLOCK_MONOTONIC";
    emlrtFunction = "emlrtClockGettimeMonotonic";
else
    coder.internal.assert(clockname == "realtime", 'Coder:builtins:Explicit', ...
        'Unsupported clock name: ' + clockname);
    posix_clock = "CLOCK_REALTIME";
    emlrtFunction = "emlrtClockGettimeRealtime";
end
info = struct('posix_clock', posix_clock, 'emlrtFunction', emlrtFunction);