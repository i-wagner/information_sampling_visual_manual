function stimulusCoordinates = getStimCoord(exper, logCol, logFiles)

    % Wrapper function
    % Extracts sorted stimulus coordinates of participants in conditions
    %
    % NOTE:
    % Stimulus locations are also stored in log files, however, those ones
    % are not sorted, i.e., individual stimuli do no have a fixed position
    % in the vector and it is not clear how vector entries map to stimuli
    % from easy/difficult sets. This function brings stimulus coordinates
    % in an order
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
    % logFiles:
    % cell-matrix; log files of participants in conditions
    %
    % Output
    % stimulusCoordinates:
    % cell-matrix; positions of individual stimuli (columns) in trials
    % (rows), seperately for stimulus difficulty (pages)

    %% Get stimulus coordinates
    stimulusCoordinates =cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject = exper.num.SUBJECTS(s);
            logFile = logFiles{thisSubject, c};
            if isempty(logFile)
                continue
            end
    
            % Create structure with stimulus locations
            stimulusCoordinates{thisSubject,c} = ...                  
                sortStimLoc(logFile(:,logCol.STIMULUS_POSITION_X), ...
                            logFile(:,logCol.STIMULUS_POSITION_Y), ...
                            logFile(:,logCol.N_DISTRACTOR_EASY), ...
                            logFile(:,logCol.N_DISTRACTOR_DIFFICULT), ...
                            logFile(:,logCol.N_TARGETS), ...
                            logFile(:,logCol.DIFFICULTY_TARGET));
        end
    end

end