function t = getLocalTime
%MATLAB Code Generation Private Function

% Returns a MATLAB struct emulating the output of the C localtime function
% with an added nanoseconds field for better precision:
%
%   structTm.tm_nsec
%   structTm.tm_sec
%   structTm.tm_min
%   structTm.tm_hour
%   structTm.tm_mday
%   structTm.tm_mon
%   structTm.tm_year
%   structTm.tm_isdst

% tm_year and tm_mon are offset from C by 1900 and 1 respectively to
% give real-world values for the year and month.

%   Copyright 2019 The MathWorks, Inc.
%#codegen
coder.inline("never");
if coder.internal.runs_in_matlab()
    structTm = makeEMLRTStructTm();
    coder.ceval('-barrier','-jit','emlrtWallclock',coder.wref(structTm));
    t = structTm;
elseif coder.internal.time.impl.isWin64Codegen()
    t = getLocaltimeWin64();
else
    % Default to POSIX
    timespec = callPOSIXClockGettime("realtime");
    t = timespecToLocaltime(timespec);
end

%--------------------------------------------------------------------------

function t = timespecToLocaltime(timespec)
structTm = makeStructTm();
structTm = coder.ceval('*localtime',coder.rref(timespec.tv_sec));
t = toMATLABTimespec(structTm, timespec.tv_nsec);

%--------------------------------------------------------------------------

function structTm = getLocaltimeWin64()
coder.inline('always');
word_t = coder.opaque('WORD','0','HeaderFile','<windows.h>');
systemTime = struct('wYear', word_t, ...
    'wMonth', word_t, ...
    'wDay', word_t, ...
    'wHour', word_t, ...
    'wMinute', word_t, ...
    'wSecond', word_t, ...
    'wMilliseconds', word_t);
coder.cstructname(systemTime, 'SYSTEMTIME', 'extern', 'HeaderFile', '<windows.h>');
coder.ceval('-barrier','-jit','GetLocalTime',coder.wref(systemTime));
structTm.tm_nsec = cast(systemTime.wMilliseconds, 'double')*1e6;
structTm.tm_sec = cast(systemTime.wSecond, 'double');
structTm.tm_min = cast(systemTime.wMinute, 'double');
structTm.tm_hour = cast(systemTime.wHour, 'double');
structTm.tm_mday = cast(systemTime.wDay, 'double');
structTm.tm_mon = cast(systemTime.wMonth, 'double');
structTm.tm_year = cast(systemTime.wYear, 'double');

tzInfo = coder.opaque('TIME_ZONE_INFORMATION','HeaderFile','<windows.h>');
dstInfo = int32(0);
dstInfo = coder.ceval('-barrier','-jit','GetTimeZoneInformation',coder.wref(tzInfo));
structTm.tm_isdst = dstInfo == cast(coder.opaque('DWORD','TIME_ZONE_ID_DAYLIGHT','HeaderFile','<windows.h>'), 'int32');

%--------------------------------------------------------------------------

function structTm = toMATLABTimespec(origStructTm, nsec)
% Normalize the timespec to something that contains primitive data. Since we need to use floating
% point arithmetic to compute the elapsed time, we just use double here.
structTm.tm_nsec = double(nsec);
structTm.tm_sec = double(origStructTm.tm_sec);
structTm.tm_min = double(origStructTm.tm_min);
structTm.tm_hour = double(origStructTm.tm_hour);
structTm.tm_mday = double(origStructTm.tm_mday);
structTm.tm_mon = double(origStructTm.tm_mon)+1; % +1 to make 1-based
structTm.tm_year = double(origStructTm.tm_year)+1900; % +1900 for human year
structTm.tm_isdst = logical(origStructTm.tm_isdst);

%--------------------------------------------------------------------------
