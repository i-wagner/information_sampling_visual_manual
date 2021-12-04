function timespec = callPOSIXClockGettime(clockname)
%MATLAB Code Generation Private Function

% Returns a MATLAB struct emulating the output of the POSIX clock_gettime function:
%
%   t.tv_nsec - coder.opaque('long')
%   t.tv_sec - coder.opaque('time_t')
%
% The value of CLOCKNAME should be "monotonic" or "realtime"

%   Copyright 2019 The MathWorks, Inc.
%#codegen
coder.internal.errorIf(coder.internal.runs_in_matlab(), 'Coder:builtins:Explicit', ...
    'Internal error: This function is not supported when running in MATLAB. E.g. MEX, SIM, etc. Use callEMLRTClockGettime instead');
coder.internal.errorIf(coder.internal.time.impl.isWin64Codegen(), 'Coder:builtins:Explicit', ...
    'Internal error: This function is not supported for Windows codegen');
clockInfo = getClockInfo(clockname);
timespec = getTimePOSIX(clockInfo.posix_clock);

%--------------------------------------------------------------------------

function timespec = getTimePOSIX(posix_clock)
coder.inline('always');
timespec = coder.internal.time.impl.makeTimespec();
status = coder.internal.indexInt(0);
status = coder.ceval('-barrier','clock_gettime', ...
    coder.opaque('clockid_t',posix_clock,'HeaderFile','<time.h>'), ...
    coder.wref(timespec));
checkPOSIXStatus('clock_gettime',status);

%--------------------------------------------------------------------------

function checkPOSIXStatus(fcn,status)
if status ~= zeros('like',status)
    if coder.internal.runs_in_matlab()
        % Don't query errorno to avoid JIT fallback
        coder.internal.error('Coder:toolbox:POSIXCallFailed',fcn,status);
    elseif coder.internal.hasRuntimeErrors()
        reportPOSIXError(fcn,status);
    end
end

%--------------------------------------------------------------------------

function reportPOSIXError(fcn,status)
coder.cinclude('<errno.h>');
num = cast(coder.opaque('int','errno'),coder.internal.indexIntClass());
coder.internal.error('Coder:toolbox:POSIXCallFailedErrno',fcn,status,num);

%--------------------------------------------------------------------------
