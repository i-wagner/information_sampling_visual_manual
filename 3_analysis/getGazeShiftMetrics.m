function [onsets, offsets, duration, amplitude, latency] = getGazeShiftMetrics(gazeTrace, idxGazeShifts, idxStimOn)

    % Calculcates metrics of gaze shifts.
    % The following metrics are calculates:
    % - Gaze shift onset coordinates
    % - Gaze shift offset coordinates
    % - Gaze shift durations
    % - Gaze shift amplitudes
    % - Gaze shift latencies
    %   If gaze shifts happened before or right at stimulus onset, calculate
    %   latency based on the stimulus onset timestamp. This way we can can
    %   easily identify gaze shifts that happened before onset, since they
    %   will have a negative latency. For gaze shifts with onset right at
    %   stimulus onset the assumption is that they were likely programmed
    %   before stimulus onset
    %
    %   For the first gaze shift after stimulus onset, calculate the latency
    %   based on the timestamp of stimulus onset. For all subsequent gaze
    %   shifts, use the timetamps of the preceding gaze shift
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
    tsStimOnset = gazeTrace(idxStimOn,1);
    nGazeShifts = numel(onsets(:,1));
    
    isAfterStimOnset = false;
    latency = NaN(nGazeShifts, 1);
    for g = 1:nGazeShifts % Gaze shift
        thisOnset = onsets(g,1);
        if (thisOnset <= tsStimOnset)
            comparisonSample = tsStimOnset;
        elseif (thisOnset > tsStimOnset) & ~isAfterStimOnset
            comparisonSample = tsStimOnset;
            isAfterStimOnset = true;
        elseif (thisOnset > tsStimOnset) & isAfterStimOnset
            comparisonSample = offsets(g-1,1);
        end
        latency(g) = thisOnset - comparisonSample;
    end

end