function gaze = getGazeData(exper, screen, anal, nTrials)

    % Wrapper function
    % Extracts gaze data from gaze trace files. The following analysis
    % steps are performed:
    % - Extract gaze trace
    % - Get events
    % - Perform offline fixaiton check (only visual search experiment)
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
    % screen:
    % structure; settings of screen, on which experiment was recorded, as 
    % returned by the "settings_screen" script
    %
    % anal:
    % structure; vairous analysis settings, as returned by the
    % "settings_analysis" script
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
    gaze.events = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.trace = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.error.dataLoss = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.error.eventMissing = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.error.fixation.offline = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.timestamps.stimOn = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.timestamps.stimOff = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.gazeShifts.idx = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.gazeShifts.onsets = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.gazeShifts.offsets = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.gazeShifts.duration = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.gazeShifts.latency = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.gazeShifts.meanGazePos = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.gazeShifts.trialMap = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        thisCondition = exper.num.CONDITIONS(c);
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
            if isnan(thisSubject.nTrials)
                continue
            end

            thisSubject.gazeTrace = cell(thisSubject.nTrials, 1);
            thisSubject.events = NaN(thisSubject.nTrials, 5);
            thisSubject.error.dataLoss = NaN(thisSubject.nTrials, 1);
            thisSubject.error.eventMissing = NaN(thisSubject.nTrials, 1);
            thisSubject.error.fixation.offline = NaN(thisSubject.nTrials, 1);
            thisSubject.timestamp.stimOn = NaN(thisSubject.nTrials, 1);
            thisSubject.timestamp.stimOff = NaN(thisSubject.nTrials, 1);
            thisSubject.gazeShifts.idx = [];
            thisSubject.gazeShifts.onsets = [];
            thisSubject.gazeShifts.offsets = [];
            thisSubject.gazeShifts.duration = [];
            thisSubject.gazeShifts.latency = [];
            thisSubject.gazeShifts.meanGazePos = [];
            thisSubject.gazeShifts.trialMap = [];
            for t = 1:thisSubject.nTrials % Trial
                if any(thisCondition == [2, 3]) % Visual search
                    % Get gaze trace of participant
                    [thisSubject.gazeTrace{t}, thisSubject.error.dataLoss(t)] = ...
                        getGazeTrace(thisSubject.number, thisCondition, t, ...
                                     exper.path.DATA, screen);
    
                    % Get eye-link events
                    [thisSubject.events(t,:), thisSubject.error.eventMissing(t)] = ...
                        getEvents(thisSubject.gazeTrace{t}(:,4), anal.nExpectedEvents);
    
                    % Get timestamps of stimulus on- and offset
                    thisSubject.timestamp.stimOn(t) = ...
                        thisSubject.gazeTrace{t}(thisSubject.events(t,4),1);
                    thisSubject.timestamp.stimOff(t) = ...
                        thisSubject.gazeTrace{t}(thisSubject.events(t,5),1);
    
                    % Perform offline fixation check
                    thisSubject.error.fixation.offline(t) = ...
                        checkFixation(thisSubject.gazeTrace{t}(:,2:3), ...
                                      thisSubject.events(t,4), ...
                                      anal.fixation.checkBounds, ...
                                      anal.fixation.tolerance.DVA);
    
                    % Get gaze shifts in trial
                    thisTrial.gazeShifts.idx = getGazeShifts(thisSubject.gazeTrace{t});
    
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
                    % Get events
                    [thisSubject.error.eventMissing(t), ...
                     thisSubject.timestamp.stimOn(t), ...
                     thisSubject.timestamp.stimOff(t)] = ...
                        getEventsManualSearch(thisSubject.number, thisCondition, ...
                                              t, exper.path.DATA, ...
                                              anal.nExpectedEvents);
    
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
            gaze.events{thisSubject.number,c} = thisSubject.events;
            gaze.trace{thisSubject.number,c} = thisSubject.gazeTrace;
            gaze.error.dataLoss{thisSubject.number,c} = thisSubject.error.dataLoss;
            gaze.error.eventMissing{thisSubject.number,c} = thisSubject.error.eventMissing;
            gaze.error.fixation.offline{thisSubject.number,c} = thisSubject.error.fixation.offline;
            gaze.timestamps.stimOn{thisSubject.number,c} = thisSubject.timestamp.stimOn;
            gaze.timestamps.stimOff{thisSubject.number,c} = thisSubject.timestamp.stimOff;
            gaze.gazeShifts.idx{thisSubject.number,c} = thisSubject.gazeShifts.idx;
            gaze.gazeShifts.onsets{thisSubject.number,c} = thisSubject.gazeShifts.onsets;
            gaze.gazeShifts.offsets{thisSubject.number,c} = thisSubject.gazeShifts.offsets;
            gaze.gazeShifts.duration{thisSubject.number,c} = thisSubject.gazeShifts.duration;
            gaze.gazeShifts.latency{thisSubject.number,c} = thisSubject.gazeShifts.latency;
            gaze.gazeShifts.meanGazePos{thisSubject.number,c} = thisSubject.gazeShifts.meanGazePos;
            gaze.gazeShifts.trialMap{thisSubject.number,c} =  thisSubject.gazeShifts.trialMap;
            clear thisSubject
        end
    end

end