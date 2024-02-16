function hasFixationError = checkFixation(gazeTrace, sampleStimOnset, checkBounds, tolerance)

    % Checks if gaze deviated more from the fixation cross than a tolerance
    % value
    %
    % Input
    % gazeTrace: matrix; x and y gaze coordinates
    % sampleStimOnset: integer; sample number where stimulus onset occured
    % checkBounds: integer; boundaries of area around stimulus onset,
    % within which to check fixation
    % tolerance: float; tolerance value, above which a fixation error
    % happened
    %
    % Output
    % hasFixationError: boolean; fixation error occured?

    %% Check for fixation error
    checkArea = sampleStimOnset + checkBounds;
    gazeTraceBeforeStimuli = gazeTrace(checkArea(1):checkArea(2),:);

    hasFixationError = sum(abs(gazeTraceBeforeStimuli(:)) > tolerance) > 0;

end