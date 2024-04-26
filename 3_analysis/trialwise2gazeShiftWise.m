function variableGazeShiftWise = trialwise2gazeShiftWise(exper, anal, nTrials, trialMap, variable)

    % Transforms a variable, where entries are mapped trialwise, so its 
    % entries are mapped gaze-shift-wise.
    %
    % For example, if "variable" has n entries, with each entry
    % corresponding to one trial, this function transforms "variable" by
    % checking how many gaze shifts were made per trial, and, for each gaze
    % shift in a trial, creating a mapping to the corresponding entry in 
    % "variable".
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % anal:
    % structure; vairous analysis settings, as returned by the
    % "settings_analysis" script
    %
    % variable:
    % matrix; trialwise variable, which will be transformed in a
    % gaze-shift-wise format
    %
    % nTrials:
    % matrix; number of trials that participants completed in conditions
    %
    % trialMap:
    % matrix; trial number for each entry in the corresponding gaze shift
    % matrix
    %
    % Output
    % variableGazeShiftWise:
    % same as input, but in gaze-shift-wise format

    %% Create mapping
    variableGazeShiftWise = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
            thisSubject.trialMap = trialMap{thisSubject.number,c};
            thisSubject.variable = variable{thisSubject.number,c};
            if isnan(thisSubject.nTrials) | ...
               ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            thisSubject.variableGazeShiftWise = [];
            for t = 1:thisSubject.nTrials % Trial
                thisTrial.nGazeShifts = sum(thisSubject.trialMap == t);
                thisSubject.variableGazeShiftWise = ...
                    [thisSubject.variableGazeShiftWise; ...
                     repmat(thisSubject.variable(t), thisTrial.nGazeShifts, 1)];
                clear thisTrial
            end
            variableGazeShiftWise{thisSubject.number,c} = thisSubject.variableGazeShiftWise;
            clear thisSubject
        end
    end

end