function gazeShifts = getOnAndOffsets(gazeTrace)

    % Detect on- and offset of gaze shifts.
    % Gaze shifts are all saccades and blinks, detected in a trial.
    %
    % Input
    % gazeTrace:
    % matrix; gaze trace in trial, as returned by the "getGazeTrace"
    % function.
    %
    % NOTE: NEEDS TO BE FORMATED TO CONTAIN ONLY DATA OF RECORDED EYE
    %
    % Output
    % gazeShifts: 
    % matrix; indices of gaze shift on- (:,1) and offset (:,1) as well as
    % vector with Booleans, flagging whether a gaze shift was a saccade or
    % blink (:,3)

    %% Get onsets/offsets off all saccades in trace and assign memory
    saccade.onsets = find(diff(bitget(gazeTrace(:,4), 1)) == 1) + 1;
    saccade.offsets = find(diff(bitget(gazeTrace(:,4), 1)) == -1);
    saccades.n = numel(saccade.onsets);

    %% For each detected saccade onset, find the corresponding offset
    gazeShifts = NaN(saccades.n, 3);
    for o = 1:saccades.n % Onsets
        isBlink = false;
        thisOnset = saccade.onsets(o);
        thisOffset = NaN;

        % Find offset for the current onset
        % Offset is defined as the first offset occuring after an onset
        idx = find(saccade.offsets >= thisOnset, 1, 'first');
        if ~isempty(idx)
            thisOffset = saccade.offsets(idx);
        end

        % Check whether the saccade is actually a blink.
        % Can be determined by checking whether any dataloss occured
        % between saccade on- and offset
        if all(~isnan([thisOnset, thisOffset]))
            hasDataloss = bitget(gazeTrace(thisOnset:thisOffset,4), 2);
            if any(hasDataloss)
                isBlink = true;
            end
        end

        % Store data
        gazeShifts(o,1) = thisOnset;
        gazeShifts(o,2) = thisOffset;
        gazeShifts(o,3) = isBlink;
    end

end