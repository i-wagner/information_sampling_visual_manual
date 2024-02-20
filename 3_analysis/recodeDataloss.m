function bitFlagsRecoded = recodeDataloss(bitFlags)

    % Recodes flags, indicating dataloss, to indicate a specific type of
    % gaze shift, i.e., blink.
    %
    % Note:
    % The bit flags can code various things, ranging from saccades (bit
    % flag 1) over dataloss (bit flag 2), to blinks (bit flag 3).
    % Since pure dataloss and blinks are hard to desociate from each other
    % (the only difference is the acceleration/decelaration phase at the
    % start/end of blinks, which is absent for dataloss), we can recode
    % dataloss as blinks. This makes subsequent analysis easier, and gets
    % rid of the need to account for dataloss if we are interested in
    % detecting all types of different gaze shifts (i.e., blinks and
    % saccades)
    %
    % Input
    % bitFlags: vector; samplewise bit flags, indicating eye link events
    %
    % Output
    % bitFlagsRecoded: vector; same as input, but bit flags, marking
    % dataloss, are recoded as blinks

    %% Recode dataloss to gaze shifts
    bitFlagsRecoded = bitFlags;

    % Check bit flags for samples, which are coded as data loss (bit flag
    % 2), but not saccades (bit flag 1). For those entries, turn on bit
    % flag 1 as well, so they are detected as blinks
    idx = bitget(bitFlags, 2) & ~bitget(bitFlags, 1);
    bitFlagsRecoded(idx) = bitset(bitFlagsRecoded(idx), 1);

end