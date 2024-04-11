function trialDurations = getTrialDurations(exper, anal, nTrials, gaze)

    % Get duration of trials
    %
    % NOTE:
    % This function get trials durations of all trials, and does not
    % consider excluded trials
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % anal:
    % structure; various analysis settings, as returned by the
    % "settings_analysis" script
    %
    % nTrials:
    % matrix; number of completed trials per participant and condition
    %
    % gaze:
    % structure; gaze data of participants in conditions
    %
    % Output
    % trialDurations:
    % matrix; trial durations across subjects and conditions

    %% Get trial durations
    trialDurations = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            if ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            thisSubject.nTrials = nTrials(thisSubject.number,c);
            if isnan(thisSubject.nTrials)
                continue
            end

            thisSubject.trialDuration = NaN(thisSubject.nTrials, 1);
            for t = 1:thisSubject.nTrials % Trial
                % Unpack trial data
                thisTrial.timestamp.stimOn = ...
                    gaze.timestamps.stimOn{thisSubject.number,c}(t,:);
                thisTrial.timestamp.stimOff = ...
                    gaze.timestamps.stimOff{thisSubject.number,c}(t,:);

                % Get trial durations
                thisSubject.trialDuration(t) = ...
                    thisTrial.timestamp.stimOff - thisTrial.timestamp.stimOn;
                clear thisTrial
            end

            % Store data
            trialDurations{thisSubject.number,c} = thisSubject.trialDuration;
            clear thisSubject
        end
    end
end
