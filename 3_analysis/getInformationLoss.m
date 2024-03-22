function out = getInformationLoss(fixatedAois, isUnique, isBlink, gazeShiftDuration, include)

    % Checks whether a blink occured between two unique AOI visits, and
    % calculates the overall duration of said blink
    %
    % Input
    % fixatedAois:
    % vector; unique IDs of each visited AOI
    %
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
    % include:
    % vecotr; Booleans whether a gaze shift should be included in the
    % calculation of information loss. Reasons for EXCLUDING a gaze shift
    % for this might be, for example, that the gaze shift did not pass
    % quality check
    %
    % Output
    % out:
    % vector; time that was spent blinking during each unique AOI visit.
    % Non-unique AOI visits (i.e., consecutive gaze shifts within the same
    % AOI) are NaN. If no AOI vists occured, the output is a singular NaN

    %% Check for blinks during AOI visit
    % This function compares pairs of unique AOI visits and checks whether
    % a blink occures inbetween those visits. Thus, the function will
    % necessarily NOT check the last gaze shift in a trial, because it has,
    % by definition, no follow-up gaze shift. To account for this, we add
    % an additional "unique" flag to the vector with the unique fixations.
    % This way, the function can check also check whether a blink between
    % the last unique fixation in a trial and any subsequent fixation (if
    % there where any).
    % We don't need to check the last gaze shift, because no other gaze
    % shift happened after it.
    isUniqueExtended = isUnique;
    isUniqueExtended(end+1) = true;
    idxUnique = find(isUniqueExtended);
    nUniqueGazeShifts = numel(idxUnique) - 1;

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
            areaBlink = idxStart:idxEnd;
    
            blinkInAoiOccured = ...
                isBlink(areaBlink) & ...
                (fixatedAois(areaBlink) == fixatedAois(idxUnique(g))) & ...
                include(areaBlink);
            if any(blinkInAoiOccured)
                idxBlinkDurations = areaBlink(blinkInAoiOccured);
                adjustmentAmount(g) = ...
                    adjustmentAmount(g) + ...
                    sum(gazeShiftDuration(idxBlinkDurations));
            end
        end
    end
    out = NaN(size(fixatedAois));
    out(isUnique) = adjustmentAmount;

    % If no valid gaze shift is provided for analysis, "adjustmentAmount"
    % will be an empty array. To make data handling easier, assign an NaN
    % instead
    if isempty(out)
        out = NaN(size(fixatedAois));
    end

end