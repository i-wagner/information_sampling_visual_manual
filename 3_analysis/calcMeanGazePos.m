function meanGazePosition = calcMeanGazePos(gazeTrace, idxGazeShifts, sampleOffset)

    % Calculates the mean x and y gaze position inbetween consecutive gaze
    % shifts
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
    % Output
    % meanGazePosition:
    % matrix; mean and standard deviation of gaze position inbetween gaze
    % shifts. Columns 1 and 3 contain mean horizontal and vertical gaze 
    % positions, respectively. Columns 2 and 3 contain the corresponding
    % standard deviations (primarily useful for diagnostic purposes)

    %% Calculate mean gaze position between gaze shifts
    % For the last gaze shift in a trial we are looking at the average
    % position until the end of the trial
    nGazeShifts = size(idxGazeShifts, 1);
    nSamples = size(gazeTrace, 1);

    % To get the average gaze positions inbetween gaze shifts, we have to
    % use the gaze position AFTER the end of a gaze shift (thus, + 1), and
    % BEFORE the start of the next gaze shift (thus, - 1)
    boundaries = [(idxGazeShifts(:,2) + 1), ...
                  [(idxGazeShifts(2:end,1) - 1); nSamples]];

    meanGazePosition = NaN(nGazeShifts, 4);
    for g = 1:nGazeShifts % Gaze shift
        % If the offset of the gaze shift is missing, we assume that the 
        % gaze shift lasted until the end of the trial, so there is no 
        % reason to calculate the mean gaze position after it
        %
        % CAUTION: 
        % STANDARD DEVIATION OF ZERO CAN OCCUR IF GAZE SHIFT ENDED CLOSE 
        % TO RESPONSE AND THERE NOT MANY DATAPOINTS UNTIL THE RESPONSE
        if all(~isnan(boundaries(g,:)))
            % Check whether the upper boundary of the interval, we use
            % to calculate the mean gaze position, lies AFTER stimulus 
            % offset. It it does, clamp it to stimulus offset. Since no
            % stimuli are shown after their offset (duh) any gaze
            % position after offset is not informative.
            %
            % The correction has to happen here (and not at lines 31:32), 
            % because it is not guaranteed that the last gaze shift in a 
            % trial actually occured before stimulus offset.
            %
            % Only apply correction of the onset of the saccade occured
            % before stimulus offset, otherwise we might get gaze shifts
            % where offset occured before onset
            if (boundaries(g,1) <= sampleOffset) & ...
               (boundaries(g,2) > sampleOffset)
                boundaries(g,2) = sampleOffset;
            end
            idx = boundaries(g,1):boundaries(g,2);

            meanGazePosition(g,:) = ...
                [mean(gazeTrace(idx,2)), std(gazeTrace(idx,2)), ...
                 mean(gazeTrace(idx,3)), std(gazeTrace(idx,3))];
        end
    end
end