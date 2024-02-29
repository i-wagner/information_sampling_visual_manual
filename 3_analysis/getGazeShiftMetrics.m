function [onsets, offsets, duration, amplitude, latency] = getGazeShiftMetrics(gazeTrace, idxGazeShifts, idxStimOn)

    % Calculcates metrics of gaze shifts.
    % The following metrics are calculates:
    % - Gaze shift onset coordinates
    % - Gaze shift offset coordinates
    % - Gaze shift durations
    % - Gaze shift amplitudes
    % - Gaze shift latencies
    %
    % Input
    % gazeTrace:
    % matrix; gaze trace in trial, as returned by the "getGazeTrace"
    % function.
    %
    % idxGazeShifts:
    % matrix; indices of gaze shifts in gaze trace, as returned by
    % "getGazeShifts" function
    %
    % idxStimOn:
    % integer; index of sample, where stimulus onset occured
    %
    % NOTE:
    % all inputs have to be in the proper format, as returned by their
    % corresponding parent function.
    %
    % Output
    % different named vector, with the corresponding gaze shift metric

    %% Check for availability of on- and offsets
    hasOn = ~isnan(idxGazeShifts(:,1));
    hasOff = ~isnan(idxGazeShifts(:,2));

    %% On- and offset coordinates
    nGazeShifts = size(idxGazeShifts, 1);
    onsets = NaN(nGazeShifts, 3);
    offsets = NaN(nGazeShifts, 3);

    onsets(hasOn,:) = gazeTrace(idxGazeShifts(hasOn,1),1:3);
    offsets(hasOff,:) = gazeTrace(idxGazeShifts(hasOff,2),1:3);

    %% Gaze shift duration
    duration = offsets(:,1) - onsets(:,1);

    %% Gaze shift amplitudes
    % Calculate horizontal, vertical, and 2D amplitude (i.e., Euclidean
    % distance)
    amplitude = offsets(:,2:3) - onsets(:,2:3);
    amplitude = [amplitude, ...
                 sqrt(amplitude(:,1).^2 + amplitude(:,2).^2)];

    %% Gaze shift latency
    tsNextGazeShift = [gazeTrace(idxStimOn,1); offsets(1:(end-1),1)];

    latency = onsets(:,1) - tsNextGazeShift;

end