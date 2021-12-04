function emlrtDumpStack(rtmessage,callerMException)
%

% Copyright 2008-2018 The MathWorks, Inc.

msgstruct.identifier = rtmessage.identifier;
msgstruct.message = rtmessage.message;

moreinfo = coder.internal.moreinfo(rtmessage.identifier);
if ~isempty(moreinfo)
    msgstruct.message = sprintf('%s\n%s',msgstruct.message, moreinfo);
end

msgstruct.stack = rtmessage.stack(~arrayfun(@isUserTransparent,rtmessage.stack));

if isempty(msgstruct.stack)
    msgstruct.stack(1).file = '';
    msgstruct.stack(1).name = rtmessage.name;
    msgstruct.stack(1).line = 0;
elseif contains(msgstruct.stack(1).file, fullfile('matlab', 'lang', 'error'))
    msgstruct.stack = msgstruct.stack(2:end);
end

% This is the current stack which includes this function.
currentStack = dbstack('-completenames', 1);

if(nargin == 2)
    % Throw away our bogus error and just report the original error as
    % MATLAB was going to anyways.  Unfortunately - ERROR has some special
    % privileges. If the message comes from "error" the first message gets
    % printed differently.
    msgstruct.identifier = callerMException.identifier;
    msgstruct.message = callerMException.message;

    % To match MATLAB's stack, we need to build a frankenstack out of the
    % three stacks available.

    % This stack only includes calls within the generated MEX-function.
    coderStack = msgstruct.stack;

    % This is the FULL MATLAB stack including our callers.
    callerStack = callerMException.stack;

    % Strip off the common tail for the call to the MEX-function.
    callerStack = callerStack(1:(end - length(currentStack)));

    % This is finally the stack we will show the user.
    frankenStack = [callerStack; coderStack;currentStack];
    msgstruct.stack = frankenStack;
else
    msgstruct.stack = [msgstruct.stack; currentStack];
end

try
    tbm = coder.internal.TestBenchManager.getInstance();
    tbm.setErrorMsgStruct(msgstruct);
catch  %#ok<CTCH>
end
rethrow(msgstruct);

function res = isUserTransparent(stackFrame)
res = contains(stackFrame.file,fullfile('+coder','+internal','matlabCodegenHandle'));

% LocalWords:  frankenstack
