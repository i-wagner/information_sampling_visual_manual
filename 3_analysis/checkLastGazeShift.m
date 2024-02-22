function isOnBg = checkLastGazeShift(fixatedAois, flagBg)

    % Checks if the last gaze shift in a trial landed outside any AOI, and
    % if so, drops it
    %
    % Input
    % fixatedAois:
    % vector; IDs of fixated AOIs
    %
    % flagBg:
    % integer; flag that marks a background fixation
    %
    % Output
    % isOnBg:
    % Boolean; last gaze landed outside any AOI

    %% Check if last gaze shift in trial landed outside any AOI
    isOnBg = fixatedAois(end) == flagBg;

end