function metrics = getGazeShiftsManualSearch(thisSubject, thisCondition, thisTrial, pathData, tsStimOn)

    % Get gaze shifts in the manual search condition
    %
    % NOTE 1:
    % People in the manual search condition used arm movements, not eye
    % movements, to search. For consistence, we are nevertheless calling
    % movements in the manual search condition "gaze shifts"
    %
    % NOTE 2:
    % there are several differences between gaze shifts in the visual and
    % manual search experiment
    % - The first gaze shift in a trial has no offset 
    % - We cannot calculate the latency of the second gaze shift in a
    %   trial, because we are missing the offset-timestamp of the first
    %   gaze shift
    % - We have no actual "mean gaze position" in the manual search
    %   experiment (since we were not tracking the handposition over time).
    %   This, the "mean gaze position", stored in the corresponding value,
    %   is just the on- and offset coordinates of the corresponding gaze
    %   shift
    % - We don't have any sample numbers for the gaze shift on- and offsets
    %   in the manual search condition; instead, we are storing NaNs
    %
    % NOTE 3:
    % In order for the analysis pipeleine to work, the data format for
    % visual and manual search has to be identical. Thus, we are applying
    % all those fixes to maintain dimensions of variables
    %
    % Input
    % thisSubject: 
    % integer; ID of current subject
    %
    % thisCondition: 
    % integer; ID of current condition
    %
    % thisTrial: 
    % integer; ID of current trial
    %
    % pathData: 
    % string; path to data folder
    %
    % tsStimOn:
    % float; timestamps of stimulus offset. Required to correct broken
    % latencies of first gaze shifts in trials
    %
    % Output
    % metrics:
    % structure with fields "onsets", "offsets", "duration", "latency", 
    % "idx", and "meanGazePos"; metrics of gaze shifts (see also Note 2) 

    %% Get gaze shifts
    [~, subjectId] = ...
        getSubjectId(thisSubject, thisCondition, pathData);
    filepath = strcat(pathData, subjectId, "/", subjectId, "_saccades.csv");

    gazeShiftFile = readmatrix(filepath);
    if gazeShiftFile(1,17) > 1
        % Legacy code
        % Used to correct continously labeld trial numbers, i.e., trial
        % number carried over from single- to double-target experiment
        keyboard
%         gazeShiftFile(:,17) = gazeShiftFile(:,17) - ...
%                               min(gazeShiftFile(:,17)) + 1;
    end

    idx = gazeShiftFile(:,17) == thisTrial;
    gazeShifts = gazeShiftFile(idx,:);
    gazeShifts = gazeShifts(:,1:end-1);

    %% Get metrics
    nGazeShifts = size(gazeShifts, 1);

    metrics.onsets = gazeShifts(:,4:6);
    metrics.offsets = gazeShifts(:,7:9);
    metrics.duration = gazeShifts(:,10);
    metrics.latency = gazeShifts(:,11); 
    metrics.idx = [NaN(nGazeShifts, 2), gazeShifts(:,12)];
    metrics.meanGazePos = gazeShifts(:,13:16);

    %% Apply some fixes
    % Fix saccade flag
    % In the old pipeline, saccades and blink were coded as "1" and "2",
    % respectively. In the new pipeline, we use a Boolean to code whether 
    % a gaze shift was a blink. We are recoding the values from Jan's files
    % to account for this change in the new pipeline
    metrics.idx = metrics.idx == 2;

    % Fix missing latency for first gaze shift
    % For some reason, the first gaze shift in trials of the manual search
    % condition has the timestamp of its onset stored as its latency. We
    % are fixing this by calculating the actually latency, using the
    % timestamp of stimulus onset
    if ~isempty(gazeShifts)
        metrics.latency(1) = metrics.onsets(1) - tsStimOn;
    end

end