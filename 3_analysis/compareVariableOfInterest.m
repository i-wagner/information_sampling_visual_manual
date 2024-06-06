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

    %% Reshape input
    % Bring it in a shape so it can be easily compared to the data from the
    % old pipeline
    if strcmp(variableOfInterest, "proportionEasyChoices") | ...
       strcmp(variableOfInterest, "regression") | ...
       strcmp(variableOfInterest, "propGsOnChosen") | ...
       strcmp(variableOfInterest, "propGsOnSmaller")
        newPipeline = [newPipeline(:,:,1), newPipeline(:,:,2)];
    elseif strcmp(variableOfInterest, "propGsOnClosest")
        newPipeline = [newPipeline(:,:,1), newPipeline(:,:,2), ...
                       newPipeline(:,:,3), newPipeline(:,:,4)];
    elseif strcmp(variableOfInterest, "gainAbsolut")
        newPipeline = [newPipeline(:,:,1,1), newPipeline(:,:,2,1), ...
                       newPipeline(:,:,1,3), newPipeline(:,:,2,3)];
    elseif strcmp(variableOfInterest, "predPropChoicesEasy")
        newPipeline = [newPipeline(:,:,1), newPipeline(:,:,3)];
    elseif strcmp(variableOfInterest, "predPerformance")
        newPipeline = [newPipeline(:,1), newPipeline(:,3)];
    elseif strcmp(variableOfInterest, "propGsOnChosenModel")
        newPipeline = [newPipeline(:,:,2), newPipeline(:,:,4)];
    end

    % Some measures in the old pipeline had no entries for excluded
    % participants. To account for this, we drop those for the new
    % pipeleine as well
    if strcmp(variableOfInterest, "propGsOnChosen") | ...
       strcmp(variableOfInterest, "propGsOnSmaller") | ...
       strcmp(variableOfInterest, "propGsOnClosest")
        newPipeline = newPipeline(all(~isnan(newPipeline), 2),:);
    end

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
        elseif strcmp(variableOfInterest, "planningTimeEasy")
            thisVariable = thisData.sacc.time.mean.planning(:,:,2);
        elseif strcmp(variableOfInterest, "planningTimeDifficult")
            thisVariable = thisData.sacc.time.mean.planning(:,:,3);
        elseif strcmp(variableOfInterest, "inspectionTimeEasy")
            thisVariable = thisData.sacc.time.mean.inspection(:,:,2);
        elseif strcmp(variableOfInterest, "inspectionTimeDifficult")
            thisVariable = thisData.sacc.time.mean.inspection(:,:,3);
        elseif strcmp(variableOfInterest, "responseTimeEasy")
            thisVariable = thisData.sacc.time.mean.decision(:,:,2);
        elseif strcmp(variableOfInterest, "responseTimeDifficult")
            thisVariable = thisData.sacc.time.mean.decision(:,:,3);
        elseif strcmp(variableOfInterest, "proportionEasyChoices")
            thisVariable = thisData.stim.propChoice.easy(:,:,2)';
        elseif strcmp(variableOfInterest, "regression")
            thisVariable = thisData.model_io.reg.fit;
        elseif strcmp(variableOfInterest, "propGsOnChosen")
           thisVariable = thisData.sacc.propGs.onChosen_avg(:,:,2);
        elseif strcmp(variableOfInterest, "propGsOnSmaller")
            thisVariable = thisData.sacc.propGs.onSmaller_avg(:,:,2);
        elseif strcmp(variableOfInterest, "propGsOnClosest")
            thisVariable = [thisData.sacc.propGs.onCloser_avg(:,:,1), ...
                            thisData.sacc.propGs.onCloser_avg(:,:,2)];
        elseif strcmp(variableOfInterest, "propGsOnChosenModel")
            thisVariable = thisData.sacc.propGs.onAOI_modelComparision_chosenNot_ss(:,:,2);
        elseif strcmp(variableOfInterest, "nonSearchTimeEasy")
            thisVariable = thisData.sacc.time.mean.non_search(:,:,2);
        elseif strcmp(variableOfInterest, "nonSearchTimeDifficult")
            thisVariable = thisData.sacc.time.mean.non_search(:,:,3);
        elseif strcmp(variableOfInterest, "gainAbsolut")
            thisVariable = thisData.model_io.model.gain(:,:,:,3);
            thisVariable = [thisVariable(:,:,1), thisVariable(:,:,2)];
        elseif strcmp(variableOfInterest, "predPropChoicesEasy")
            thisVariable = thisData.model_io.model.choices_perfect(:,:,3);
        elseif strcmp(variableOfInterest, "empGainPerTime")
            thisVariable = [thisData.model_io.data.single.perf, ...
                            thisData.model_io.data.double.perf];
        elseif strcmp(variableOfInterest, "predPerformance")
            thisVariable = thisData.model_io.model.perf_perfect(:,3);
        elseif strcmp(variableOfInterest, "latencyTimecourse")
            thisVariable = squeeze(thisData.sacc.latency.firstGs(:,1,:));
        end
        oldPipeline = [oldPipeline, thisVariable];
    end
    
    %% Compare pipelines
    % Round to decimals to avoid false-alarms when comparing pipelines
    pipelineResultsMatch = isequaln(round(newPipeline, 14), round(oldPipeline, 14));
    if ~pipelineResultsMatch
        warning("Results from old and new pipeline do not match!");
        keyboard
    end

end