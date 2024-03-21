function [inspectionTime, dwellTimes] = getInspectionTime(fixatedAoi, targetIds, bgId, timestampOffset, adjustmentAmount, leavingTimes, useTargets)

    % Calculates dwell-times of fixation and inspection time
    %
    % NOTE 1:
    % Dwell times are defined as the time between gaze entering and leaving
    % an AOI. Inspection time is the average dwell time over all fixations
    %
    % NOTE 2:
    % Dwell times are NOT calculated for gaze shifts that landed on
    % the background and for the last gaze shift in a trial. If a blink was
    % detected during an AOI visit, we subtract the blink duration from the
    % respective dwell time to account for the lass in information during
    % the AOI visit
    %
    % Output
    % dwellTimes:
    % vector; dwell time of each unique AOI fixation.
    %
    % inspectionTime:
    % float; average dwell time across all AOI fixations

    %% Calculate dwell times and inspection time
    % Only works if we have at least one fixation, i.e., if participants
    % made at least one gaze shift to something other than the background
    useFixation = (fixatedAoi ~= bgId);
    if ~isempty(useFixation)
        if ~useTargets
            useFixation = useFixation & ...
                          all(fixatedAoi ~= targetIds, 2);        
        end
        useFixation(end) = false; % Last fixation is not considered by default
    end
    nGazeShifts = numel(fixatedAoi);
    dwellTimes = NaN(nGazeShifts, 1);
    dwellTimes(useFixation) = leavingTimes(useFixation) - ...
                              timestampOffset(useFixation) - ...
                              adjustmentAmount(useFixation);
    inspectionTime = mean(dwellTimes, 'omitnan');

end