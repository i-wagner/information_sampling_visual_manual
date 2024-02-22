function adjustmentAmount = getInformationLoss(isUnique, isBlink, gazeShiftDuration)

    % Checks whether a blink occured between two unique AOI visits, and
    % calculates the overall duration of said blink
    %
    % Input
    % isUnique:
    % vector; Booleans whether a fixation was unqiue or was followed by
    % another fixation that landed on the same AOI as the previous one
    %
    % isBlink:
    % vector; Booleans whether a fixation is a blink or not
    %
    % gazeShiftDuration:
    % vector: durations of gaze shifts
    %
    % Output
    % adjustmentAmount:
    % vector; time that was spent blinking during an unique AOI visit

    %% Check for blinks during AOI visit
    % This function compares pairs of unique AOI visits and checks whether
    % a blink occures inbetween those visits. Thus, the function will
    % necessarily NOT check the last gaze shift in a trial, because it has,
    % by definition, no follow-up gaze shift. To account for this, we add
    % an additional "unique" flag to the vector with the unique fixations.
    % This way, the function can check also check whether a blink between
    % the last unique fixation in a trial and any subsequent fixation (if
    % there where any)
    isUniqueExtended = isUnique;
    isUniqueExtended(end+1) = true;

    idxUnique = find(isUniqueExtended);
    idxBlink = find(isBlink);
    nUniqueGazeShifts = numel(idxUnique) - 1;

    % We don't need to check the last gaze shift, because no other gaze
    % shift happened after it
    adjustmentAmount = zeros(nUniqueGazeShifts, 1);
    for g = 1:nUniqueGazeShifts % Unique gaze shifts
        % Skip check check if no non-unique gaze shift occured between two
        % unique gaze shifts.
        %
        % NOTE:
        % if no gaze shift occured after the last unique gaze shift in a
        % trial, this check would evaluate to zero, because the last unique
        % gaze shift in a trial and the addeded "gaze shift" would occur
        % immediately after each other
        if diff([idxUnique(g), idxUnique(g+1)]) ~= 1
            idxStart = idxUnique(g) + 1;
            idxEnd = idxUnique(g+1) - 1;
    
            blinkOccured = any(isBlink(idxStart:idxEnd));
            if blinkOccured
                inArea = ismember(idxBlink, [idxStart, idxEnd]);
                idxBlinkDurations = idxBlink(inArea);
                adjustmentAmount(g) = ...
                    adjustmentAmount(g) + ...
                    sum(gazeShiftDuration(idxBlinkDurations));
            end
        end
    end

end