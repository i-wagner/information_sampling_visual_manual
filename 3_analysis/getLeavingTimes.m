function leavingTimes = getLeavingTimes(isBlink, timestampOffset, timestampOnset, timestampResponse)

    % Get timestamps of when a gaze shift left an AOI
    %
    % NOTE:
    % If an AOI was entered left via blink, we define the leaving time as
    % the onset of the leaving blink. If  an AOI was left via a saccade, we
    % define leaving time as the offset of the leaving saccade. This is
    % done due to differences in information uptake during blinks and
    % saccades.
    %
    % NOTE 2:
    % For the last gaze shift in a trial, we take the time at which a
    % response was placed as the leaving time.
    %
    % Input
    % isBlink:
    % vector; fixation is blink?
    %
    % timestampOffset:
    % vector; timestampts of gaze shift offsets
    %
    % timestampOnset:
    % vector; timestampts of gaze shift onsets
    %
    % timestampResponse:
    % integer; timestamp of response
    %
    % Output
    % leavingTimes:
    % vector; timestamps of when the AOI around a fixated stimulus was
    % left, according to our definition (see notes)

    %% Get leaving times
    nGazeShifts = size(isBlink, 1);
    leavingTimes = NaN(nGazeShifts, 1);
    for g = 1:(nGazeShifts - 1) % Unique AOI fixations
        if ~isBlink(g) % AOI left via saccade
            leavingTimes(g) = timestampOffset(g+1);
        elseif isBlink(g) % AOI left via blink
            leavingTimes(g) = timestampOnset(g+1);
        end
    end
    leavingTimes(end) = timestampResponse;

end