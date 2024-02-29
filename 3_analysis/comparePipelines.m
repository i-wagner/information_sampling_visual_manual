function comparePipelines(thisSubject, thisTrial, exper, logCol, s, c, t)

    %% Recreate the datastructure of the old pipeline, using the result of the new pipeline
    % If no gaze shift was detected, create an empty double array with the
    % same dimensions as in the old pipeline (just creating an emoty array
    % with [] won't work, since it is dimensionless)
    nGs = sum(thisTrial.gazeShifts.subset);
    newPipeline = double.empty(0, 29);
    if nGs > 0
        newPipeline = [ ...
            thisTrial.gazeShifts.idx(thisTrial.gazeShifts.subset,1) - thisTrial.events(4) + 1, ... Sample # onset
            thisTrial.gazeShifts.idx(thisTrial.gazeShifts.subset,2) - thisTrial.events(4) + 1, ... Sample # offset
            zeros(nGs, 1), ... Exclude gaze shift?
            thisTrial.gazeShifts.onsets(thisTrial.gazeShifts.subset,1), ... Timestamp onset
            thisTrial.gazeShifts.onsets(thisTrial.gazeShifts.subset,2), ... x-coordinate onset
            thisTrial.gazeShifts.onsets(thisTrial.gazeShifts.subset,3), ... y-coordinate onset
            thisTrial.gazeShifts.offsets(thisTrial.gazeShifts.subset,1), ... Tmestamp offset
            thisTrial.gazeShifts.offsets(thisTrial.gazeShifts.subset,2), ... x-coordinate offset
            thisTrial.gazeShifts.offsets(thisTrial.gazeShifts.subset,3), ... y-coordinat eoffset
            thisTrial.gazeShifts.duration(thisTrial.gazeShifts.subset), ... Gaze shift duration
            thisTrial.gazeTrace(thisTrial.gazeShifts.idx(thisTrial.gazeShifts.subset,1),1) - thisTrial.gazeTrace(thisTrial.events(4),1), ... Gaze shift latency (relative to stimulus onset)
            (thisTrial.gazeShifts.idx(thisTrial.gazeShifts.subset,3) + 1), ... Saccade or blink?
            thisTrial.gazeShifts.meanGazePos(thisTrial.gazeShifts.subset,1), ... Mean x-pos.
            thisTrial.gazeShifts.meanGazePos(thisTrial.gazeShifts.subset,2), ... Std x-pos.
            thisTrial.gazeShifts.meanGazePos(thisTrial.gazeShifts.subset,3), ... Mean y-pos.
            thisTrial.gazeShifts.meanGazePos(thisTrial.gazeShifts.subset,4), ... Std y-pos.
            thisTrial.gazeShifts.fixatedAois(thisTrial.gazeShifts.subset,1), ... Unique IDs
            thisTrial.gazeShifts.fixatedAois(thisTrial.gazeShifts.subset,2), ... Group IDs
            thisTrial.gazeShifts.informationLoss, ... Information loss due to blinks
            thisTrial.time.dwell, ... Dwell times
            thisTrial.gazeShifts.wentToClosest, ... Gaze shif to closest stimulus?
            (zeros(nGs, 1) + thisSubject.logFile(t,logCol.N_DISTRACTOR_EASY)), ... n easy distractors
            (zeros(nGs, 1) + thisTrial.chosenTarget.response), ... Chosen target
            NaN(nGs, 1), ... Timelock relative to trial start
            NaN(nGs, 1), ... Timelock relative to trial end
            (zeros(nGs, 1) + t), ... Trial number
            thisTrial.gazeShifts.distanceCurrent, ... Distance to currently fixated stimulus
            (zeros(nGs, 1) + thisSubject.logFile(t,logCol.N_DISTRACTOR_DIFFICULT)), ... n difficult distractors
            (zeros(nGs, 1) + thisTrial.nDistractorsChosenSet)]; % n distractors in chosen set

        % Add timelock, while ignoring fixations on distractors
        idxNonBgGs = newPipeline(:,18) ~= exper.stimulus.id.BACKGROUND;
        nNonBgGs = sum(idxNonBgGs);
        newPipeline(idxNonBgGs,24) = (1:nNonBgGs)';
        newPipeline(idxNonBgGs,25) = ((nNonBgGs-1):-1:0)';
    end

    %% Get data from old pipeline
    oldPipeline = load("/Users/ilja/Desktop/oldGs.mat");
    
    idx = oldPipeline.gazeShifts{s,c}(:,26) == t;
    oldPipelineTrial = oldPipeline.gazeShifts{s,c}(idx,:);

    %% Compare results
    % We round decimals, because floats cannot be compared properly
    % otherwise
    nDecimals = 11;
    pipelinesEqual = isequaln(round(newPipeline,nDecimals), ...
                              round(oldPipelineTrial,nDecimals));
    if ~pipelinesEqual
        warning("Difference in results detects, please check pipelines!");
        keyboard
    end

end