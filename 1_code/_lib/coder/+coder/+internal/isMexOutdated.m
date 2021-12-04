function result = isMexOutdated(mexFcnName)
%

%   Copyright 2017-2019 The MathWorks, Inc.

mexFcnFile = which(mexFcnName);

if exist('coder.internal.Project', 'class') ~= 8
    warning(message('EMLRT:runTime:ProfilingNoCoder'));
    result = true;
    return;
end

project = coder.internal.Project;
props = project.getMexFcnProperties(mexFcnFile);

if isempty(props) || ~isfield(props, 'ResolvedFunctions') ...
    || ~isfield(props, 'EntryPoints')
    result = true;
    return;
end

ep = props.EntryPoints;

% Check all entry point time stamp is ok
for i = 1:numel(ep)
    D = dir(ep(i).FullPath);
    % file does not exist or timestamps do not match.
    if isempty(D) || ~isequal(D.datenum, ep(i).TimeStamp)
        result = true;
        return;
    end
end

% Verify resolved functions
resolvedFunctions = props.ResolvedFunctions;
outOfDateIdx = project.verifyResolvedFunction(resolvedFunctions);

isCodeCoverage = strcmp(getenv('TESTCOVERAGE'), 'PROFILER');
if isCodeCoverage
    % If we are doing code coverage, don't complain about toolbox functions.
    for idx = 1 : numel(outOfDateIdx)
        if ~isToolboxFcn(resolvedFunctions(idx).resolved)
            result = true;
            return;
        end
    end
    result = false;
else
    result = ~isempty(outOfDateIdx);
end
end

function ret = isToolboxFcn(path)
    exp = {'matlab[\\/]toolbox'};
    if ismac
        exp{end+1} = 'MATLAB_R\d{4}[ab].app/toolbox'; 
    end
    match = regexp(path, exp, 'once');
    ret = any(cellfun(@any, match));
end