function planningTime = getPlanningTime(timestampsGazeShiftOffsets, timestampStimulusOnset)

    % Calculate planning time
    %
    % NOTE 1:
    % planning time is defined as the time between onset of the stimulus
    % display and the offset of the first gaze shift in a trial
    %
    % Input
    % timestampsGazeShiftOffsets:
    % vector; timestamps of gaze shift offsets
    %
    % timestampStimulusOnset:
    % integer; timestamp of stimulus display onset
    %
    % Output
    % planningTime:
    % integer; planning time. Is NaN if no planning time could be
    % calculated, i.e., if not gaze shift was made in a trial

    %% Calculate planning time
    nUniqueFixations = numel(timestampsGazeShiftOffsets);

    planningTime = NaN;
    if nUniqueFixations > 0
        planningTime = timestampsGazeShiftOffsets(1) - ...
                       timestampStimulusOnset;
    end

end