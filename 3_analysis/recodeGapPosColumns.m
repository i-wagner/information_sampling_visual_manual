function locationColumns  = recodeGapPosColumns(shownTarget, gapPosition)

    % Recodes columns with gap positions if one of them is missing.
    %
    % NOTE:
    % in the single-target condition, the gap position of the shown target,
    % by default, is stored in the column where we store the gap position
    % of the easy target in the double-target condition.
    %
    % CAUTION:
    % this function only has an effect for data from the single-target
    % condition, since data in the double-target condition is sorted
    % correctly to begin with
    %
    % Input
    % shownTarget:
    % vector; trialwise IDs of the target that shown in a trial
    %
    % gapPosition:
    % matrix; trialwise ID of the response key that was pressed in a trial
    % 
    % Output
    % locationColumns:
    % matrix; trialwise ID of the response key that was pressed in a trial,
    % bit with IDs placed in the correct column, depending on which target
    % was shown

    %% Recode columns with gap position if one is missing
    % No need to change anything if no column if missing (as in the
    % double-target condition)
    locationColumns = gapPosition;

    columnMissing = any(all(isnan(gapPosition), 1));
    if columnMissing
        nTrials = size(shownTarget, 1);
        
        locationColumns = NaN(nTrials, 2);
        for t = 1:nTrials % Trials
            locationColumns(t,shownTarget(t)) = gapPosition(t,1);
        end
    end

end