function [eventMissing, tsStimOn, tsStimOff] = getEventsManualSearch(thisSubject, thisCondition, thisTrial, pathData, nExpectedEvents)

    % Get sample numbers of events in manual search experiment
    %
    % NOTE:
    % Unlike for the visual search experiment, we have no sample numbers
    % for events in the manual search experiment. Instead, we only have
    % access to actual timestamps.
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
    % nExpectedEvents: 
    % integer; number of events we expect to happen in a trial
    %
    % Output
    % eventMissing: 
    % boolean; any events missing?
    %
    % tsStimOn:
    % float; timestamp of stimulus onset
    %
    % tsStimOff:
    % float; timestamp of stimulus offset

    %% Get events
    [~, subjectId] = ...
        getSubjectId(thisSubject, thisCondition, pathData);
    filepath = strcat(pathData, subjectId, "/", subjectId, "_events.csv");
    eventFile = readmatrix(filepath);

    eventMissing = false;
    events = eventFile(thisTrial,:)';
    tsStimOn = events(4);
    tsStimOff = events(5);
    if numel(events) ~= nExpectedEvents
        eventMissing = true;
        tsStimOn = NaN;
        tsStimOff = NaN;
    end

end