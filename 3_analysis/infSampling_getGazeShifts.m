function events_gs = infSampling_getGazeShifts(gazeTrace, ts_stimOnset, minDur_gs, screen_x, screen_y)

    % Detect all gaze shifts in a trial. We define a gaze shift as a
    % saccade or a blink
    % Input
    % gazeTrace:     matrix, containing gaze trace in trial. Rows are
    %                datasamples, columns have to contain to following data
    %                (:, 1):   timestamps
    %                (:, 2:3): x- and y-coordinates
    %                (:, 4):   bit, marking events in the trial
    % ts_stimOnset:  timestamp of stimulus onset
    % minDur_gs:     minimum duration of saccades; if a gaze shift is
    %                shorter we flag it for exclusion
    % screen_:       size of screen (deg); if a gaze shift ended somewhere
    %                outside the screen, we flag it for exclusion
    % Output
    % events_gs:     matrix with the parameters of each gaze shift, detected
    %                in a trial
    %                (:, 1:2):     range of datasamples, spanned by gaze shift
    %                (:, 3:5):     timestampe as well as x- and y-coordinates
    %                              of gaze shift onset
    %                (:, 6:8):     timestampe as well as x- and y-coordinates
    %                              of gaze shift offset
    %                (:, 9):       gaze shift duration
    %                (:, 10):      gaze shift latency
    %                (:, 11):      flag, marking if a gaze shift was caused
    %                              by a saccade (1) or a blink (2)
    %                (:, [12 14]): mean x and y gaze position after gaze
    %                              shift, until next gaze shift (or until
    %                              end of trial, if it is the last gaze
    %                              shift in a trial)
    %                (:, [13 15]): standard deviation x and y gaze position 
    %                              after gaze shift, until next gaze shift 
    %                              (or until end of trial, if it is the
    %                              last gaze shift in a trial)

    %% Find all saccade on- and offsets in gaze trace
    sacc_onsets  = find(diff(bitget(gazeTrace(:, 4), 1)) == 1) + 1;
    sacc_offsets = find(diff(bitget(gazeTrace(:, 4), 1)) == -1);
    sacc_onOff   = [];
    if ~isempty(sacc_onsets) && ~isempty(sacc_offsets)

        % For each detected onset, determine an offset
        no_saccInTrial = size(sacc_onsets, 1);
        sacc_onOff     = [sacc_onsets NaN(no_saccInTrial, 1) zeros(no_saccInTrial, 1)];
        for s = 1:no_saccInTrial % Saccades in trial

            bitToCheck = 1;
            sacc_onOff(s, 2) = ...
                infSampling_eventOffset(sacc_onOff(s, 1), sacc_offsets, gazeTrace(:, 4), bitToCheck);
            clear bitToCheck

        end
        clear s no_saccInTrial

        % Add saccade on- and offset timestamps/coordinates, saccade
        % duration, saccade latency and a flag, marking an entry in the
        % matrix as a saccade
        sacc_onOff(:, end+1:end+3) = gazeTrace(sacc_onOff(:, 1), 1:3);     % Onset timestamp and x-/y-coordinates
        if isnan(sacc_onOff(end, 2))                                       % Offset timestamp and x-/y-coordinates

            li_offAvailable = ~isnan(sacc_onOff(:, 2));

            sacc_onOff(:, end+1:end+3) = [gazeTrace(sacc_onOff(li_offAvailable, 2), 1:3); ...
                                          NaN(1, 3)];
            clear li_offAvailable

        else

            sacc_onOff(:, end+1:end+3) = gazeTrace(sacc_onOff(:, 2), 1:3);

        end
        sacc_onOff(:, end+1) = sacc_onOff(:, 2) - sacc_onOff(:, 1);        % Saccade duration    
        sacc_onOff(:, end+1) = sacc_onOff(:, 4) - ts_stimOnset;            % Saccade latency
        sacc_onOff(:, end+1) = 1;                                          % Flag, marking a saccade

    end
    clear sacc_onsets sacc_offsets


    %% Find all blink on- and offsets in gaze trace
    % Blinks were detected in one of two ways: they are either lead by and
    % followed by a saccade, in which case we already detected them as a
    % saccade in the previous section. Alternatively, they can occur as
    % "pure" blinks, in which case they are not lead/followed by a
    % saccade. If a blink was led on by a saccade, we adjust the blink
    % on-/offset, so it corresponds to the saccade's on-/offset, and we
    % remove the false saccade from the saccade matrix
    blink_onsets  = find(diff(bitget(gazeTrace(:, 4), 2)) == 1) + 1;
    blink_offsets = find(diff(bitget(gazeTrace(:, 4), 2)) == -1);
    blink_onOff   = [];
    if ~isempty(blink_onsets)

        % For each detected onset, determine an offset
        no_blinkInTrial = size(blink_onsets, 1);
        blink_onOff     = [blink_onsets NaN(no_blinkInTrial, 1) zeros(no_blinkInTrial, 1)];
        for b = 1:no_blinkInTrial % Blink in trial

            bitToCheck = 2;
            blink_onOff(b, 2) = ...
                infSampling_eventOffset(blink_onOff(b, 1), blink_offsets, gazeTrace(:, 4), bitToCheck);
            clear bitToCheck

        end
        clear b no_blinkInTrial

        % For each detected blink, check if it is part of a saccade. If so,
        % replace its indices with the indices of the corresponding saccade
        % (since this is the true on-/offset of the blink) and remove the
        % indices from the saccade matrix (since they represent a blink,
        % which was erroneously detected as a saccade)
        no_blinkInTrial = size(blink_onOff, 1);
        no_saccInTrial  = size(sacc_onOff, 1);
        no_ds           = size(gazeTrace, 1);
        saccBlink_idx   = [];
        for b = 1:no_blinkInTrial % Blink in trial

            % If a blink offset is missing, we assume the blink lasted
            % until the end of a trial
            if all(~isnan(blink_onOff(b, :)))

                blink_trace = (blink_onOff(b, 1):blink_onOff(b, 2))';

            else

                blink_trace = (blink_onOff(b, 1):no_ds)';

            end

            for s = 1:no_saccInTrial % Saccade in trial

                % If a saccade offset is missing, we assume the saccade lasted
                % until the end of a trial
                if all(~isnan(sacc_onOff(s, :)))

                    sacc_trace = (sacc_onOff(s, 1):sacc_onOff(s, 2))';

                else

                    sacc_trace = (sacc_onOff(s, 1):no_ds)';

                end
                if all(ismember(blink_trace, sacc_trace))

                    blink_onOff(b, 1:2) = sacc_onOff(s, 1:2);
                    saccBlink_idx       = [saccBlink_idx; s];

                end
                clear sacc_trace

            end
            clear s blink_trace

        end
        clear b no_blinkInTrial no_saccInTrial

        % Exclude saccades that are actually blinks and repeating blinks;
        % the latter case sometimes happens if multiple blinks happen
        % during a saccade
        sacc_onOff(saccBlink_idx, :) = [];

        [~, idx_unique, ~] = unique(blink_onOff(:, 1));
        blink_onOff        = blink_onOff(idx_unique, :);
        clear  saccBlink_idx idx_unique

        % Add flag, marking an entry in the matrix as a blink
        blink_onOff(:, end+1:end+3) = gazeTrace(blink_onOff(:, 1), 1:3);
        if isnan(blink_onOff(end, 2))

            li_offAvailable = ~isnan(blink_onOff(:, 2));

            blink_onOff(:, end+1:end+3) = [gazeTrace(blink_onOff(li_offAvailable, 2), 1:3); ...
                                           NaN(1, 3)];
            clear li_offAvailable

        elseif any(isnan(blink_onOff(1:end-1, 2)))

            keyboard
            blink_onOff(:, end)         = [];
            blink_onOff(:, end+1:end+3) = gazeTrace(blink_onOff(:, 2), 1:3);

        else

            blink_onOff(:, end+1:end+3) = gazeTrace(blink_onOff(:, 2), 1:3);

        end
        blink_onOff(:, end+1) = blink_onOff(:, 2) - blink_onOff(:, 1);
        blink_onOff(:, end+1) = NaN;
        blink_onOff(:, end+1) = 2;

    end
    clear blink_onsets blink_offsets


    %% Construct gaze shift matrix
    % Put all detected gaze shifts in one matrix and sort by their respective
    % onset time
    events_gs = sortrows([sacc_onOff; blink_onOff], 1);
    clear sacc_onOff blink_onOff


    %% Determine mean gaze position before/after each gaze shift
    no_gs = size(events_gs, 1);
    no_ds = size(gazeTrace, 1);

    gazePose_mean = NaN(no_gs, 4);
    for gs = 1:no_gs % Gaze shift

        % Get indices for interval between end of current gaze shift and
        % begin of next gaze shift. For the last gaze shift, we look at the
        % gaze position until the end of the trial
        idx_endCurrGs = events_gs(gs, 2) + 1;
        if gs == no_gs

            idx_startNextGs = no_ds;

        else

            idx_startNextGs = events_gs(gs+1, 1) - 1;

        end

        % Get mean and standard deviation of gaze position between end of
        % current gaze shift and begin of next gaze shift. If the offset of
        % the gaze shift is missing, we assume that the gaze shift lasted
        % until the end of the trial, so there is no reason to calculate
        % the mean gaze position after it
        % CAUTION: STANDARD DEVIATION OF ZERO CAN OCCUR IF GAZE SHIFT ENDED
        % CLOSE TO RESPONSE AND THERE NOT MANY DATAPOINTS UNTIL THE RESPONSE
        if all(~isnan([idx_endCurrGs idx_startNextGs]))

            gazePose_mean(gs, :) = [mean(gazeTrace(idx_endCurrGs:idx_startNextGs, 2)) ... x-position
                                    std(gazeTrace(idx_endCurrGs:idx_startNextGs, 2)) ...
                                    mean(gazeTrace(idx_endCurrGs:idx_startNextGs, 3)) ... y-position
                                    std(gazeTrace(idx_endCurrGs:idx_startNextGs, 3))];
            clear idx_endCurrGs idx_startNextGs

        end

    end
    events_gs = [events_gs gazePose_mean];
    clear gs no_gs no_ds gazePose_mean


    %% Check for very short gaze shift, incomplete gaze shifts and gaze shifts with coordinates outside the screen
    % Very short gaze shifts sometimes occur after a "pure" blink, which was
    % not led/followed by a saccade; we treat those very short gaze shifts
    % as false-positives and exclude them later. Incomplete gaze shifts
    % sometimes occur if a participant decides to give a response while a
    % gaze shift is still in flight. Gaze shifts with on-/offsets outside
    % the screen sometimes occur for who knows what reason. Gaze shifts
    % might have an extrem on-/offset, which can be clearily attributed as
    % dataloss or a blink; we keep those cases, since we have at least some
    % clue what happened there
    li_excld = events_gs(:, 10) < minDur_gs | ...                                  Gaze shift too short
               isnan(events_gs(:, 2)) | ...                                        Offset missing
               abs(events_gs(:, 5))  > screen_x & abs(events_gs(:, 5))  < 90 | ... Onset/offset coordinate outside screen
               abs(events_gs(:, 8))  > screen_x & abs(events_gs(:, 8))  < 90 | ...
               abs(events_gs(:, 13)) > screen_x & abs(events_gs(:, 13)) < 90 | ...
               abs(events_gs(:, 6))  > screen_y & abs(events_gs(:, 6))  < 90 | ...
               abs(events_gs(:, 9))  > screen_y & abs(events_gs(:, 9))  < 90 | ...
               abs(events_gs(:, 15)) > screen_y & abs(events_gs(:, 15)) < 90;

    events_gs(:, 3) = events_gs(:, 3) + li_excld;
    clear li_excld

end