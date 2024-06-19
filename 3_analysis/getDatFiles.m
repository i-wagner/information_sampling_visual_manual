function gaze = getDatFiles(exper, screen, anal, nTrials)

    % Wrapper function
    % Processes dat files and extracts relevamt data from them. The
    % following analysis steps are performed:
    % - Extract gaze traces from dat files
    % - Extract sample numbers eye link events from dat files
    % - Extract timestamps of stimulus on- and offset   
    % - Perform offline fixaiton check, using the gaze traces, stored in
    %   the dat files (only visual search experiment)
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

    %% Process dat filrs
    gaze.events = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.trace = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.error.dataLoss = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.error.eventMissing = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.error.fixation.offline = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.timestamps.stimOn = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    gaze.timestamps.stimOff = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        thisCondition = exper.num.CONDITIONS(c);
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
            if isnan(thisSubject.nTrials) | ...
               ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            thisSubject.gazeTrace = cell(thisSubject.nTrials, 1);
            thisSubject.events = NaN(thisSubject.nTrials, 5);
            thisSubject.error.dataLoss = false(thisSubject.nTrials, 1);
            thisSubject.error.eventMissing = false(thisSubject.nTrials, 1);
            thisSubject.error.fixation.offline = false(thisSubject.nTrials, 1);
            thisSubject.timestamp.stimOn = NaN(thisSubject.nTrials, 1);
            thisSubject.timestamp.stimOff = NaN(thisSubject.nTrials, 1);
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
                elseif any(thisCondition == [4, 5]) % Manual search
                    % Get events
                    [thisSubject.error.eventMissing(t), ...
                     thisSubject.timestamp.stimOn(t), ...
                     thisSubject.timestamp.stimOff(t)] = ...
                        getEventsManualSearch(thisSubject.number, thisCondition, ...
                                              t, exper.path.DATA, ...
                                              anal.nExpectedEvents);
                end
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
            clear thisSubject
        end
    end

end