function [events, eventMissing] = getEvents(eventFlags, nExpectedEvents)

    % Get sample numbers of eye-link events.
    %
    % Input
    % eventFlags: vector; event flag for each gaze sample. Events have to
    % coded as bit flags, where a the third bit corresponds to an eye-link
    % events
    % nExpectedEvents: integer; number of events we expect to happen in a
    % trial
    %
    % Output
    % events: vector; sample number of all eye-link events in a trial
    % eventMissing: boolean; any events missing?

    %% Get events
    eventMissing = false;
    events = find(bitget(eventFlags, 3));
    if numel(events) ~= nExpectedEvents
        eventMissing = true;
        events = NaN(nExpectedEvents, 1);
    end

end