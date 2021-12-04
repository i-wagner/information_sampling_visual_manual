function t = getTime()
%MATLAB Code Generation Private Function

%   Copyright 2018-2019 The MathWorks, Inc.

%#codegen
clockname = "monotonic";
if coder.internal.runs_in_matlab()
    t = callEMLRTClockGettime(clockname);
elseif coder.internal.time.impl.isWin64Codegen()
    % No need to call toMATLABTimespec since we don't go through POSIX timespec struct
    t = getTimeWin64();
else
    % Default to POSIX
    t = toMATLABTimespec(callPOSIXClockGettime(clockname));
end

%--------------------------------------------------------------------------

function timespec = getTimeWin64()
coder.inline('always');
coder.cinclude('<windows.h>','JITSupported');
persistent freq;
if isempty(freq)
    freqL = makeLargeInteger();
    status = coder.internal.indexInt(0);
    status = coder.ceval('QueryPerformanceFrequency',coder.wref(freqL));
    checkWindowsStatus('QueryPerformanceFrequency',status);
    freq = cast(freqL.QuadPart,'double');
end
timeL = makeLargeInteger();
status = coder.internal.indexInt(0);
status = coder.ceval('-barrier','QueryPerformanceCounter',coder.wref(timeL));
checkWindowsStatus('QueryPerformanceCounter',status);
BILLION = 1e9;
timeDouble = cast(timeL.QuadPart,'double');
timeRemainder = mod(timeDouble,freq);
seconds = (timeDouble - timeRemainder) / freq;
nanoseconds = (timeRemainder * BILLION) / freq;
timespec.tv_sec = seconds;
timespec.tv_nsec = nanoseconds;

%--------------------------------------------------------------------------

function L = makeLargeInteger()
L = struct('QuadPart',coder.opaque('LONGLONG','0','HeaderFile','<windows.h>'));
coder.cstructname(L,'LARGE_INTEGER','extern','HeaderFile','<windows.h>');

%--------------------------------------------------------------------------

function timespec = toMATLABTimespec(origTimespec)
% Normalize the timespec to something that contains primitive data. Since we need to use floating
% point arithmetic to compute the elapsed time, we just use double here.
timespec.tv_sec = double(origTimespec.tv_sec);
timespec.tv_nsec = double(origTimespec.tv_nsec);

%--------------------------------------------------------------------------

function checkWindowsStatus(fcn,status)
if status == zeros('like',status)
    coder.internal.error('Coder:toolbox:WindowsCallFailed',fcn);
end

%--------------------------------------------------------------------------
