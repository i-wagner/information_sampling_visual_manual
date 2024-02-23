function atLeastOneAoiGazeShift = checkOneAoiGazeShift(fixatedAoiIds, flagBg)

    % Check whether at least one gaze shift was made to any AOI
    %
    % Input
    % fixatedAoiIds:
    % vector; IDs of fixated AOIs. Can be either group IDs or unique IDs,
    % as long background fixations can be clearly identified
    %
    % flagBg:
    % integer; flag that marks background fixations
    %
    % Output
    % atLeastOneAoiGazeShift:
    % Boolean; was at least one AOI fixated?

    %% Perform check
    hasMadeGazeShifts = ~isempty(fixatedAoiIds);
    oneGazeShiftToAoi = ~all(fixatedAoiIds == flagBg);

    atLeastOneAoiGazeShift = false;
    if hasMadeGazeShifts & oneGazeShiftToAoi
        atLeastOneAoiGazeShift = true;
    end

end