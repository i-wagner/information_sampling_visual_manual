function results = projectWhich(projectPath, callContexts, interrupt)
%

% Copyright 2005-2017 The MathWorks, Inc.

    results = {};
    originalDir = pwd();
    restorePath = onCleanup(@() cd(originalDir));
    
    try
        cd(projectPath);
        iterator = callContexts.iterator();
        i = 1;
        while (iterator.hasNext())
            if (interrupt.get())
                return;
            end
            
            pair = iterator.next();
            caller = char(pair.getFirst());
            callee = char(pair.getSecond());
            
            s = which('-all', callee, 'in', caller);
            
            results{i}{1} = pair; %#ok<AGROW>
            results{i}{2} = findValidResult(s); %#ok<AGROW>
            
            i = i + 1;
        end
    catch
        % Do nothing.
    end

end

function [resolved] = findValidResult(results)
resolved = '';
validPaths = results(contains(results, filesep)); % Retain only results that contain paths.
userFcns = getUserFcns(validPaths); % Prioritize user functions over toolbox ones.
if ~isempty(userFcns)
    resolved = userFcns{1};
elseif ~isempty(validPaths)
    resolved = validPaths{1};
end 
end

function [bool] = getUserFcns(results)
bool = results(~contains(results, matlabroot));
end