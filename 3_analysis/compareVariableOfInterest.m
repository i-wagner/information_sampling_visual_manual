function compareVariableOfInterest(newPipeline, variableOfInterest, suffix)

    % Compares content of a variable, generated using the new analysis 
    % pipeline, to it's counterpart, generated using the old pipeline
    %
    % Input
    % newPipeline:
    % varying type; data from new pipeleine, which to compare to the same
    % data from the old pipeline
    %
    % variableOfInterest:
    % string; dependent variable of interest:
    % "proportionValid"
    % "proportionTrialsWithResponse"
    %
    % suffix:
    % string; use results of old pipeline where only trials
    % ("_withExclusion") or where trials and subjects where excluded 
    % ("_allExclusions"). Use "" for no exclusions
    %
    % Output
    % --

    %% Get variable of interest from old pipeline
    conditionLabels = ["oldGs_visual", "oldGs_manual"];
    oldPipeline = [];
    for c = 1:2 % Condition
        pathToData = strcat("/Users/ilja/Dropbox/12_work/", ...
                            "mr_informationSamplingVisualManual/2_data/", ...
                            conditionLabels{c}, suffix, ".mat");
        thisData = load(pathToData);
        if strcmp(variableOfInterest, "proportionValid")
            thisVariable = thisData.exper.prop.val_trials;
        elseif strcmp(variableOfInterest, "proportionTrialsWithResponse")
            thisVariable = thisData.exper.prop.resp_trials;
        elseif strcmp(variableOfInterest, "timeLostExcldTrials")
            thisVariable = thisData.exper.timeLostExcldTrials;
        elseif strcmp(variableOfInterest, "aoiFix")
            thisVariable = thisData.sacc.propGs.aoiFix_mean;
        elseif strcmp(variableOfInterest, "propCorrectEasy")
            thisVariable = thisData.perf.hitrates(:,:,2);
        elseif strcmp(variableOfInterest, "propCorrectDifficult")
            thisVariable = thisData.perf.hitrates(:,:,3);
        end
        oldPipeline = [oldPipeline, thisVariable];
    end
    
    %% Compare pipelines
    % Round to decimals to avoid false-alarms when comparing pipelines
    pipelineResultyMatch = isequaln(round(newPipeline, 14), round(oldPipeline, 14));
    if ~pipelineResultyMatch
        warning("Results from old and new pipeleine do not match!");
        keyboard
    end

end