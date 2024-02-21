function isUnique = getUniqueFix(idFixatedAoi)

    % Get unique fixations, i.e., AOI visists.
    % Sometimes participants make small, corrective gaze shifts within
    % AOIs; we are not interested in those, so we flag them for later
    % exclusion
    %
    % Input
    % idFixatedAoi:
    % vector; unique IDs of fixated AOIs.
    %
    % NOTE:
    % input as to be UNIQUE IDs, not group IDs. The latter are, by
    % definition, not unique, so we cannot use them to get unique gaze
    % shifts.
    %
    % Output
    % isUnique:
    % vector; unique or consecutive AOI fixation?

    %% Remove consecutive gaze shifts
    % No idea why we do not keep consecutive gaze shifts on the background
    % (since we cannot really if those changed the current fixation
    % location or not). This is the method we used for the first
    % publication, so we stick with it
    isUnique = diff([zeros(1); idFixatedAoi]) ~= 0;

end