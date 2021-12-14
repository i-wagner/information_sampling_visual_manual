function curr_off = infSampling_eventOffset(curr_on, all_off, bitTrace, bitToCheck)

    % Determines the event offset for a given event onset. An event can be
    % a saccade or a blink
    % Input
    % curr_on:    integer, representing the samplenumber of an event onset
    %             for which we want to find a corresponding offset
    % all_off:    column vector with sample numbers of all detected event
    %             offsets in trial
    % bitTrace:   column vector, with bit for each sample.
    % bitToCheck: to determine if an offset truely belongs to a given
    %             onset, we check if the gaze trace between onset and
    %             offset is continous. For this, we use the bit, which
    %             either indicates a saccade (bit == 1) or a blink
    %             (bit == 2). If the provided bit is on for all datapoints
    %             between onset and offset, onset and offset belong to the
    %             same type of gaze shift
    % Output
    % curr_off:   integer, representing the the samplenumber of an event
    %             offset, belonging to the input "curr_on"

    %% Find event offset, belonging event onset "curr_on"
    idx_ev_offset = find(all_off >= curr_on, 1, 'first');
    if ~isempty(idx_ev_offset) % Offset detected

        curr_off = all_off(idx_ev_offset);

    else                       % Not offset detected

        curr_off = NaN;

    end


    %% Check if onset and offset belong to the same event
    % If so, the bit, marking a given event (blink or saccade) being in
    % progress, should be continously turned on between onset and the
    % determined offset
    if ~isnan(curr_off) % Offset detected

        ev_trace_bit = bitget(bitTrace(curr_on:curr_off), bitToCheck);
        if any(ev_trace_bit ~= 1)

            keyboard

        end
        clear idx_ev_offset ev_trace_bit

    end

end