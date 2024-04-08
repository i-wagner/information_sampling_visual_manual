function comparePipelines(thisSubject, thisTrial, exper, logCol, s, c, t, suffix)

    %% Recreate the datastructure of the old pipeline, using the result of the new pipeline
    % If no gaze shift was detected, create an empty double array with the
    % same dimensions as in the old pipeline (just creating an emoty array
    % with [] won't work, since it is dimensionless)
    nGs = sum(thisTrial.gazeShifts.subset);
    newPipeline = double.empty(0, 29);
    if nGs > 0
        % In the manual search experiment, some values are defined slightly
        % different compared to the values from the visual search experiment
        if ismember(c, [1, 2])
            % - Sample number of on- and offset are expressed relative to
            %   stimulus onset, because this is how we did it in the old
            %   pipeline
            % - Latency is expressed relative to stimulus offset, because
            %   this is how we did it in the old pipeleine
            sampleOn = thisTrial.gazeShifts.idx(thisTrial.gazeShifts.subset,1) - thisTrial.events(4) + 1;
            sampleOff = thisTrial.gazeShifts.idx(thisTrial.gazeShifts.subset,2) - thisTrial.events(4) + 1;
            latency = thisTrial.gazeTrace(thisTrial.gazeShifts.idx(thisTrial.gazeShifts.subset,1),1) - ...
                      thisTrial.gazeTrace(thisTrial.events(4),1);
            yOn = thisTrial.gazeShifts.onsets(thisTrial.gazeShifts.subset,3);
            yOff = thisTrial.gazeShifts.offsets(thisTrial.gazeShifts.subset,3);
            meanY = thisTrial.gazeShifts.meanGazePos(thisTrial.gazeShifts.subset,3);
        elseif ismember(c, [3, 4])
            % Gaze coordinates are centered on the stimulus center again,
            % because that is how we did it in the old pipeline
            sampleOn = thisTrial.gazeShifts.onsets(thisTrial.gazeShifts.subset,1);
            sampleOff = thisTrial.gazeShifts.offsets(thisTrial.gazeShifts.subset,1);
            latency = thisTrial.gazeShifts.latency(thisTrial.gazeShifts.subset);
            yOn = thisTrial.gazeShifts.onsets(thisTrial.gazeShifts.subset,3) + exper.fixation.location.y.DVA;
            yOff = thisTrial.gazeShifts.offsets(thisTrial.gazeShifts.subset,3) + exper.fixation.location.y.DVA;
            meanY = thisTrial.gazeShifts.meanGazePos(thisTrial.gazeShifts.subset,3) + exper.fixation.location.y.DVA;
        end
        newPipeline = [ ...
            sampleOn, ... Sample # onset
            sampleOff, ... Sample # offset
            zeros(nGs, 1), ... Exclude gaze shift?
            thisTrial.gazeShifts.onsets(thisTrial.gazeShifts.subset,1), ... Timestamp onset
            thisTrial.gazeShifts.onsets(thisTrial.gazeShifts.subset,2), ... x-coordinate onset
            yOn, ... y-coordinate onset
            thisTrial.gazeShifts.offsets(thisTrial.gazeShifts.subset,1), ... Tmestamp offset
            thisTrial.gazeShifts.offsets(thisTrial.gazeShifts.subset,2), ... x-coordinate offset
            yOff, ... y-coordinate offset
            thisTrial.gazeShifts.duration(thisTrial.gazeShifts.subset), ... Gaze shift duration
            latency, ... Gaze shift latency (relative to stimulus onset)
            (thisTrial.gazeShifts.idx(thisTrial.gazeShifts.subset,3) + 1), ... Saccade or blink?
            thisTrial.gazeShifts.meanGazePos(thisTrial.gazeShifts.subset,1), ... Mean x-pos.
            thisTrial.gazeShifts.meanGazePos(thisTrial.gazeShifts.subset,2), ... Std x-pos.
            meanY, ... Mean y-pos.
            thisTrial.gazeShifts.meanGazePos(thisTrial.gazeShifts.subset,4), ... Std y-pos.
            thisTrial.fixatedAois.uniqueIds(thisTrial.gazeShifts.subset), ... Unique IDs
            thisTrial.fixatedAois.groupIds(thisTrial.gazeShifts.subset), ... Group IDs
            thisTrial.gazeShifts.informationLoss(thisTrial.gazeShifts.subset), ... Information loss due to blinks
            thisTrial.time.dwell(thisTrial.gazeShifts.subset), ... Dwell times
            thisTrial.gazeShifts.wentToClosest(thisTrial.gazeShifts.subset), ... Gaze shift to closest stimulus?
            (zeros(nGs, 1) + thisSubject.logFile(t,logCol.N_DISTRACTOR_EASY)), ... n easy distractors
            (zeros(nGs, 1) + thisTrial.chosenTarget.response), ... Chosen target
            NaN(nGs, 1), ... Timelock relative to trial start
            NaN(nGs, 1), ... Timelock relative to trial end
            (zeros(nGs, 1) + t), ... Trial number
            thisTrial.gazeShifts.distanceCurrent(thisTrial.gazeShifts.subset), ... Distance to currently fixated stimulus
            (zeros(nGs, 1) + thisSubject.logFile(t,logCol.N_DISTRACTOR_DIFFICULT)), ... n difficult distractors
            (zeros(nGs, 1) + thisTrial.nDistractorsChosenSet)]; % n distractors in chosen set

        % Add timelock, while ignoring fixations on distractors
        idxNonBgGs = newPipeline(:,18) ~= exper.stimulus.id.BACKGROUND;
        nNonBgGs = sum(idxNonBgGs);
        newPipeline(idxNonBgGs,24) = (1:nNonBgGs)';
        newPipeline(idxNonBgGs,25) = ((nNonBgGs-1):-1:0)';
    end

    %% Get data from old pipeline
    if ismember(c, [1, 2])
        filename = "oldGs_visual";
        idxCol = c;
    elseif ismember(c, [3, 4])
        filename = "oldGs_manual";
        switch c
            case 3
                idxCol = 1;
            case 4
                idxCol = 2;
        end
    end
    pathToData = strcat("/Users/ilja/Dropbox/12_work/", ...
                        "mr_informationSamplingVisualManual/2_data/", ...
                        filename, suffix, ".mat");
    oldPipeline = load(pathToData);
    if strcmp(suffix, "_withExclusion")
        oldPipeline = oldPipeline.sacc;
    end

    idx = oldPipeline.gazeShifts{s,idxCol}(:,26) == t;
    oldPipelineTrial = oldPipeline.gazeShifts{s,idxCol}(idx,:);
    if ismember(c, [3, 4])
        % When the data was exported from the old pipeline, the badTrial
        % was exclusion was not implemented, so we need to account for this
        % here
        if ismember(t, thisSubject.badTrials)
            oldPipelineTrial = double.empty(0, 29);
        end
    end

    %% Compare results
    % We round decimals, because floats cannot be compared properly
    % otherwise
    nDecimals = 9;
    pipelinesEqual = isequaln(round(newPipeline, nDecimals), ...
                              round(oldPipelineTrial, nDecimals));

    % In the old pipeline, whether a gaze shift went to the closest stimulus
    % relative to fixation was determined by using the subset of gaze shifts
    % (i.e., after excluding gaze shifts that did not pass the quality check).
    % In some cases, this resulted in an inaccurate result, because the current
    % fixation location was determined wrongly after excluding gaze shifts 
    % (e.g., a gaze shift was made before stimulus onset, which was not caught 
    % properly, or the initial gaze shift to an AOI was excluded, while a 
    % consecutive gaze shift was included). In
    % the new pipeline, this analysis is conducted using all gaze shifts.
    wentToClosestMismatch = ~isequaln(round(newPipeline(:,21), nDecimals), ...
                                      round(oldPipelineTrial(:,21), nDecimals));
    restMatch = isequaln(round(newPipeline(:,[1:20, 22:end]), nDecimals), ...
                         round(oldPipelineTrial(:,[1:20, 22:end]), nDecimals));
    if wentToClosestMismatch & restMatch
        pipelinesEqual = true;
    end     

    % There are some special rules for a handful of participants, mostly
    % due to the new gaze shift detection algorithm we are using for the
    % new analysis pipeline
    %
    % c: 1; s: 6; t: 118
    % The old pipeline detects an additional gaze shift for this
    % participant, which the new pipeline does not detect. This is because
    % the new pipeline labels the gaze shift in question as being part of a
    % blink, while the old pipeline misses to do so
    %
    % c: 1; s: 10; t: 85
    % The old pipeline detects two additional gaze shifts for this
    % participant, which the new pipeline excludes. This is because the old
    % pipeleine missdetects the offset of one of the gaze shifts in
    % question, which causes it to be included (the new pipeleine detects
    % the correct offset). Due to this inclusion, another subsequent gaze
    % shift is also not dropped, even though it should be dropped (since
    % its mean gaze position is labeled as NaN, and no AOI is determined
    % for it)
    %
    % c: 1; s: 12; t: 34
    % The old pipeline misses one gaze shift, which the new pipeline
    % detects. This is because the old pipeline looks for gaze shifts only
    % between stimulus on- and offset; since the gaze shift in question
    % occurs at the same sample as stimulus onset, the old pipeline misses
    % it (but detects the offset, which, however, has no corresponding
    % onset).
    %
    % c: 1; s: 12; t: 107
    % Missed gaze shift with onset right on stimulus onset sample 
    % (see "c: 1; s: 12; t: 34")
    %
    % c: 1; s: 19; t: 59
    % The new pipeline includes two additional gaze shifts, which the old
    % pipeline excludes. Additionally, the new pipeline excludes a gaze
    % shift, which the old one includes. This is because, in the old 
    % pipeline, sampels with dataloss are labeld as "2", while they are 
    % labeled as "3" in the new pipeline. This causes the saccade detection 
    % in the old pipeleine to missdetect offsets, whenever the traces 
    % transitions from a saccade ("1") to a dataloss label ("2"); this does 
    % not happen in the new pipeline. As a consequence, whether a saccade
    % is consecutive (which, in turn, depends on the immediately preceeding
    % and succeeding saccade) might change depending on the pipeline that
    % we use.
    %
    % c: 1; s: 19; t: 94
    % New pipeline includes a gaze shift that the old one excludes, and new
    % pipeline excludes a gaze shift that the new one includes (see 
    % "c: 1; s: 19; t: 59")
    %
    % c: 2; s: 4; t: 122
    % Missed gaze shift with onset right on stimulus onset sample 
    % (see "c: 1; s: 12; t: 34")
    %
    % c: 2; s: 5; t: 36
    % Missed gaze shift with onset right on stimulus onset sample 
    % (see "c: 1; s: 12; t: 34")
    %
    % c: 2; s: 12; t: 245
    % The old analysis pipeline stores a gaze shift, which the new one does 
    % not store. If participants, in the old pipeline, did not make any 
    % gaze shifts, we created a vector, with the first 18 columns being 
    % NaN; in the new pipeline, the corresponding vector is just empty.
    % This is the reason for the additional gaze shift in the old pipeline.
    %
    % c: 2; s: 12; t: 247
    % Placeholder gaze shift in trial without any gaze shift (see 
    % "c: 2; s: 12; t: 245")
    %
    % c: 2; s: 12; t: 248
    % Placeholder gaze shift in trial without any gaze shift (see 
    % "c: 2; s: 12; t: 245")
    %
    % c: 2; s: 18; t: 156
    % Missed gaze shift with onset right on stimulus onset sample 
    % (see "c: 1; s: 12; t: 34")
    %
    % c: 2; s: 19; t: 23
    % The new pipeline detects two gaze shifts, which the old one does not
    % detect. This is due to the missdetection of gaze shift offset (see
    % "c: 1; s: 19; t: 59")
    %
    % c: 2; s: 19; t: 76
    % Missmatch in detected gaze shifts between old and new pipeline. DID
    % NOT CHECK WHY THIS HAPPENS, but most likely due to missdetected gaze
    % shift offsets in old pipeleine (see "c: 1; s: 19; t: 59")
    %
    % c: 2; s: 19; t: 84
    % Placeholder gaze shift in trial without any gaze shift (see 
    % "c: 2; s: 12; t: 245")
    %
    % c: 3; s: 13; t: 11
    % One gaze is flagged as going to the closest stimulus in the old
    % pipeline, but not in the new pipeline. This is because we re-center
    % gaze and stimulus coordinates in the new pipeleine, which creates
    % some rounding differences. Since this particular trial actually
    % contains two stimuli that are closest to gaze, the old pipeleine
    % chooses the first minimum, while the new pipeleine think ones minimum
    % is smaller then the other (because of rounding) and chooses this one
    %
    % c: 4; s: 7; t: [34:35, 37:39, 151:153]:
    % Placeholder gaze shift in trial without any gaze shift (see 
    % "c: 2; s: 12; t: 245")
    %
    % c: 4; s: 8; t: 14:
    % Placeholder gaze shift in trial without any gaze shift (see 
    % "c: 2; s: 12; t: 245")
    if ~pipelinesEqual & ...
       ~(c == 1 & s == 6 & t == 118) & ...
       ~(c == 1 & s == 10 & t == 85) & ...
       ~(c == 1 & s == 12 & t == 34) & ...
       ~(c == 1 & s == 12 & t == 107) & ...
       ~(c == 1 & s == 19 & t == 59) & ...
       ~(c == 1 & s == 19 & t == 94) & ...
       ~(c == 2 & s == 4 & t == 122) & ...
       ~(c == 2 & s == 5 & t == 36) & ...
       ~(c == 2 & s == 12 & t == 245) & ...
       ~(c == 2 & s == 12 & t == 247) & ...
       ~(c == 2 & s == 12 & t == 248) & ...
       ~(c == 2 & s == 18 & t == 156) & ...
       ~(c == 2 & s == 19 & t == 23) & ...
       ~(c == 2 & s == 19 & t == 76) & ...
       ~(c == 2 & s == 19 & t == 84) & ...
       ~(c == 3 & s == 13 & t == 11) & ...
       ~(c == 3 & s == 19 & t == 92) & ...
       ~(c == 4 & s == 7 & any(t == [34:35, 37:39, 151:153]))
        warning("Difference in results detects, please check pipelines!");
        keyboard
    end

end