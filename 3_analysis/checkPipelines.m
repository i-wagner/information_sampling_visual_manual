function checkPipelines(exper, logCol, logFiles, gaze, fixations, time, choice)

    % Wrapper function
    % Compares results from old to results from new pipeleine. 
    %
    % NOTE:
    % This function only check the "gazeShifts" variable from the old
    % pipeleine; no ther variable are checked between the pipelines
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    % 
    % logCol:
    % structure; column indices for log files, as returned by the
    % "settings_log" script
    % 
    % gaze:
    % structure; gaze data of participants in conditions, as returned by
    % the "getGazeData" function
    % 
    % fixations:
    % structure; fixated AOIs across participants and conditions, as
    % returned by the "getFixatedAois" function
    % 
    % time:
    % structure; time-related variables across participants and conditions,
    % as returned by the "getTimes" function
    % 
    % choice:
    % structure; choice-related variables across participants and
    % conditions, as returned by the "getChoices" function
    %
    % Ouput
    % --

    %% Check results from old and new pipeline
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.logFile = logFiles.files{thisSubject.number,c};
            thisSubject.nTrials = logFiles.nCompletedTrials(thisSubject.number,c);
            if isempty(thisSubject.logFile)
                continue
            end
            for t = 1:thisSubject.nTrials % Trial
                % Unpack data
                thisTrial.idx = ...
                    gaze.gazeShifts.trialMap{thisSubject.number,c} == t;
                thisTrial.gazeShifts.meanGazePos = ...
                    gaze.gazeShifts.meanGazePos{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.gazeShifts.idx = ...
                    gaze.gazeShifts.idx{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.gazeShifts.onsets = ...
                    gaze.gazeShifts.onsets{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.gazeShifts.offsets = ...
                    gaze.gazeShifts.offsets{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.gazeShifts.duration = ...
                    gaze.gazeShifts.duration{thisSubject.number,c}(thisTrial.idx);
                thisTrial.gazeShifts.latency = ...
                    gaze.gazeShifts.latency{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.timestamp.stimOn = ...
                    gaze.timestamps.stimOn{thisSubject.number,c}(t);
                thisTrial.timestamp.stimOff = ...
                    gaze.timestamps.stimOff{thisSubject.number,c}(t);
                thisTrial.events = ...
                    gaze.events{thisSubject.number,c}(t,:);
                thisTrial.gazeTrace = ...
                    gaze.trace{thisSubject.number,c}{t,:};
                thisTrial.gazeShifts.subset = ...
                    logical(fixations.subset{thisSubject.number,c}(thisTrial.idx));
                thisTrial.gazeShifts.informationLoss = ...
                    fixations.informationLoss{thisSubject.number,c}(thisTrial.idx);
                thisTrial.fixatedAois.groupIds = ...
                    fixations.fixatedAois.groupIds{thisSubject.number,c}(thisTrial.idx);
                thisTrial.fixatedAois.uniqueIds = ...
                    fixations.fixatedAois.uniqueIds{thisSubject.number,c}(thisTrial.idx);
                thisTrial.gazeShifts.wentToClosest = ...
                    fixations.wentToClosest{thisSubject.number,c}(thisTrial.idx);
                thisTrial.gazeShifts.distanceCurrent = ...
                    fixations.distanceCurrent{thisSubject.number,c}(thisTrial.idx);
                thisTrial.time.dwell = ...
                    time.dwell{thisSubject.number,c}(thisTrial.idx);
                thisTrial.chosenTarget.response = ...
                    choice.target{thisSubject.number,c}(t);
                thisTrial.nDistractorsChosenSet = ...
                    choice.nDistractorsChosenSet{thisSubject.number,c}(t);
    
                % Compare pipelines
                comparePipelines(thisSubject, thisTrial, exper, logCol, s, c, t);
                clear thisTrial
            end
        end
        clear thisSubject
    end
end
