function responseTime = getResponseTime(fixatedAoiIds, timestampsGazeShiftOffset, timestampResponse, flagTarget, flagBg)

    % Calculate response time
    %
    % NOTE 1:
    % Response time is defined as the time between offset of the last gaze
    % shift in a trial and the time when the response was placed
    %
    % NOTE 2:
    % The decision time can only be calculated if a participant's last
    % gaze shift either landed on a target or, if it did not land on a
    % target, if gaze shifts only landed on the background after the last
    % gaze shift to a target
    %
    % Input
    % fixatedAoiIds:
    % vector; IDs of fixated AOIs. Can be either group IDs or unique IDs,
    % as long as target fixations are clearly identifiable
    %
    % timestampsGazeShiftOffset:
    % vector; timestamps of gaze shift offsets
    %
    % timestampResponse:
    % integer; timestamp of when response was placed
    %
    % flagTarget:
    % vector; IDs that identify target fixations
    %
    % flagBg:
    % integer; ID that identifies background fixation
    %
    % Output
    % responseTime:
    % integer; response time

    %% Calculate response time
    responseTime = NaN;
    if ~isempty(fixatedAoiIds)
        targetFixations = any(fixatedAoiIds == flagTarget, 2);
        idxLastTargetFixation = find(targetFixations, 1, 'last');
        if ~isempty(idxLastTargetFixation)
            isSubsequentTargetFixation = ...
                any(fixatedAoiIds(idxLastTargetFixation:end) == flagTarget, 2);
            isSubsequentBgFixation = ...
                fixatedAoiIds(idxLastTargetFixation:end) == flagBg;
            if all(isSubsequentTargetFixation | isSubsequentBgFixation)
                responseTime = timestampResponse - ...
                               timestampsGazeShiftOffset(idxLastTargetFixation);
            end
        end
    end

end