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

    %% Add parameters for detected saccades
    % Only calculate parameter for gaze shifts, for which both, on- and
    % offset are available
    isComplete = all(~isnan(idxGazeShifts(:,1:2)), 2);
    idxCompleteGazeShifts = idxGazeShifts(isComplete,:);

    %% On- and offset coordinates
    onsets = gazeTrace(idxCompleteGazeShifts(:,1),1:3);
    offsets = gazeTrace(idxCompleteGazeShifts(:,2),1:3);

    %% Gaze shift duration
    duration = gazeTrace(idxCompleteGazeShifts(:,2),1) - ...
               gazeTrace(idxCompleteGazeShifts(:,1),1);

    %% Gaze shift amplitudes
    % Calculate horizontal, vertical, and 2D amplitude (i.e., Euclidean
    % distance)
    amplitude = gazeTrace(idxCompleteGazeShifts(:,2),2:3) - ...
                gazeTrace(idxCompleteGazeShifts(:,1),2:3);
    amplitude = [amplitude, ...
                 sqrt(amplitude(:,1).^2 + amplitude(:,2).^2)];

    %% Gaze shift latency
    timeStampsOnsets = [gazeTrace(idxStimOn,1); ...
                        gazeTrace(idxCompleteGazeShifts(1:end-1,2),1)];
    latency = gazeTrace(idxCompleteGazeShifts(:,1),1) - timeStampsOnsets;

end