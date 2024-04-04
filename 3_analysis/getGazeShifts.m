function gazeShifts = getGazeShifts(exper, gaze, nTrials)

    % Wrapper function
    % Extracts gaze data from gaze trace files. The following analysis
    % steps are performed:
    % - Get gaze shifts in trial
    % - Calculate gaze shift metrics
    % - Re-center gaze shift coordinates to location of fixation cross
    %   (only manual search experiment)
    %
    % NOTE:
    % In the manual search condition, gaze shift offsets can sometimes be
    % missing. If this happens for the first gaze shift in a trial, this
    % trial likely contained pen-dragging (participants never lifted the
    % pen, but dragged it across the screen). If the offset of the last
    % gaze shit is missing, participants likely lifted their arm before
    % placing a response (this way, an offset would be recorded but no
    % offset, since the arm was never lowered again).
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % nTrials:
    % matrix; number of trials that participants completed in conditions
    %
    % Output
    % gaze:
    % structure; gaze data of participants in conditions

    %% Get gaze data
    % In the manual search experiment, events and "gaze shifts" are stored 
    % in a dedicated  file, instead of being coded in the gaze trace file 
    % (since there is none).
    %
    % We call the movements in the manual search experiments "gaze shifts"
    % to be consistent, although this is rather unintuitive
    gazeShifts.idx = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gazeShifts.onsets = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gazeShifts.offsets = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gazeShifts.duration = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gazeShifts.latency = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gazeShifts.meanGazePos = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gazeShifts.trialMap = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        thisCondition = exper.num.CONDITIONS(c);
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
            thisSubject.gazeTrace = gaze.trace{thisSubject.number,c};
            thisSubject.events = gaze.events{thisSubject.number,c};
            thisSubject.timestamp.stimOn = gaze.timestamps.stimOn{thisSubject.number,c};
            if isnan(thisSubject.nTrials)
                continue
            end

            thisSubject.gazeShifts.idx = [];
            thisSubject.gazeShifts.onsets = [];
            thisSubject.gazeShifts.offsets = [];
            thisSubject.gazeShifts.duration = [];
            thisSubject.gazeShifts.latency = [];
            thisSubject.gazeShifts.meanGazePos = [];
            thisSubject.gazeShifts.trialMap = [];
            for t = 1:thisSubject.nTrials % Trial
                if any(thisCondition == [2, 3]) % Visual search
                    % Get gaze shift on- and offsets in trial
                    thisTrial.gazeShifts.idx = ...
                        getOnAndOffsets(thisSubject.gazeTrace{t});
    
                    % Calculate gaze shift metrics
                    [thisTrial.gazeShifts.onsets, thisTrial.gazeShifts.offsets, ...
                     thisTrial.gazeShifts.duration, ~, thisTrial.gazeShifts.latency] = ...
                        getGazeShiftMetrics(thisSubject.gazeTrace{t}, ...
                                            thisTrial.gazeShifts.idx, ...
                                            thisSubject.events(t,4));
                    thisTrial.gazeShifts.meanGazePos = ...
                        calcMeanGazePos(thisSubject.gazeTrace{t}, ...
                                        thisTrial.gazeShifts.idx, ...
                                        thisSubject.events(t,5));
                elseif any(thisCondition == [4, 5]) % Manual search
                    % Get gaze shifts
                    thisTrial.gazeShifts = ... 
                        getGazeShiftsManualSearch(thisSubject.number, ...
                                                  thisCondition, t, ...
                                                  exper.path.DATA, ...
                                                  thisSubject.timestamp.stimOn(t));
    
                    % Adjust gaze shift coordinates
                    % Degree-of-visual-angle coordinates can be expressed in 
                    % different reference frames, e.g., relative to the 
                    % fixation cross position or relative to the screen center. 
                    % In the VISUAL SEARCH experiment, they are expressed 
                    % relative to the fiaxtion cross, while in the MANUAL 
                    % SEARCH experiment, they are expressed relative to screen 
                    % center. Here, we correct coordinates in the manual search 
                    % experiment so they are in line with how coordinates
                    % are expressed in the visual search experiment
                    thisTrial.gazeShifts.onsets(:,3) = ...
                        adjustVerticalCoordinates(thisTrial.gazeShifts.onsets(:,3), ...
                                                  exper.fixation.location.y.DVA);
                    thisTrial.gazeShifts.offsets(:,3) = ...
                        adjustVerticalCoordinates(thisTrial.gazeShifts.offsets(:,3), ...
                                                  exper.fixation.location.y.DVA);
                    thisTrial.gazeShifts.meanGazePos(:,3) = ...
                        adjustVerticalCoordinates(thisTrial.gazeShifts.meanGazePos(:,3), ...
                                                  exper.fixation.location.y.DVA);
                end

                % Store trial data
                thisSubject.gazeShifts.idx = ...
                    [thisSubject.gazeShifts.idx; thisTrial.gazeShifts.idx];
                thisSubject.gazeShifts.onsets = ...
                    [thisSubject.gazeShifts.onsets; thisTrial.gazeShifts.onsets];
                thisSubject.gazeShifts.offsets = ...
                    [thisSubject.gazeShifts.offsets; thisTrial.gazeShifts.offsets];
                thisSubject.gazeShifts.duration = ...
                    [thisSubject.gazeShifts.duration; thisTrial.gazeShifts.duration];
                thisSubject.gazeShifts.latency = ...
                    [thisSubject.gazeShifts.latency; thisTrial.gazeShifts.latency];
                thisSubject.gazeShifts.meanGazePos = ...
                    [thisSubject.gazeShifts.meanGazePos; thisTrial.gazeShifts.meanGazePos];
                thisSubject.gazeShifts.trialMap = ...
                    [thisSubject.gazeShifts.trialMap; ...
                     zeros(numel(thisTrial.gazeShifts.duration), 1) + t];
                clear thisTrial
            end

            % Store subject data
            gazeShifts.idx{thisSubject.number,c} = thisSubject.gazeShifts.idx;
            gazeShifts.onsets{thisSubject.number,c} = thisSubject.gazeShifts.onsets;
            gazeShifts.offsets{thisSubject.number,c} = thisSubject.gazeShifts.offsets;
            gazeShifts.duration{thisSubject.number,c} = thisSubject.gazeShifts.duration;
            gazeShifts.latency{thisSubject.number,c} = thisSubject.gazeShifts.latency;
            gazeShifts.meanGazePos{thisSubject.number,c} = thisSubject.gazeShifts.meanGazePos;
            gazeShifts.trialMap{thisSubject.number,c} =  thisSubject.gazeShifts.trialMap;
            clear thisSubject
        end
    end

end