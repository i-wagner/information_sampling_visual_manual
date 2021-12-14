function stay_aoi_all = infSampling_getFixationsAOI(gazeTrace, ts_stimOnset, stimLoc_x, stimLoc_y, aoi_radius, debug_plot_allStim)

    % Determines which AOIs where fixated in a trial, which event caused
    % the eye to land within the AOI (saccade, blink, etc.) and how long
    % the eye remained within the AOI. If an AOI was entered/left via a
    % drift or if the AOI was not left at all, dwell time is calculated
    % with the respective time of entry or the time of response.
    % Input
    % gazeTrace:     four-column matrix, containing gaze trace in trial, with
    %                timestamps (:, 1), x- (:, 2) and y-coordinates (:, 3) 
    %                and bit (:, 4)
    % ts_stimOnset:  timestamp of stimulus onset
    % stimLoc_x/_y:  one-row vector, with each column containing the x- and
    %                y-coordinats of each stimulus, shown in a trial;
    %                not-shown stimuli should be NaN
    % aoi_radius:    radius of AOI around stimulus center; whenever the
    %                eyes land within the area around the stimulus, described
    %                by the radius, we count this as the stimulus being
    %                fixated
    % debug_:        create plots of the location of all stimuli and the
    %                endpoints of each detected saccade
    % Output
    % stay_aoi_all:  matrix with each AOI visit in a trial
    %                (:, 1):     index of visited stimulus in location matrix
    %                (:, 2:3):   datasamples during which AOI was visited
    %                (:, 4:11):  datasamples, offset time, offset coordinates,
    %                            duration, latency and type (1 == saccade, 
    %                            2 == blink, 3 == drift) of gaze shift that
    %                            entered aoi
    %                (:, 12:19): same structure, but for gaze shift that
    %                            left AOI
    %                (:, 20):    dwell time in AOI (offset time of gaze
    %                            shift entering AOI - offset time of gaze
    %                            shift leaving AOI)
    %                (:, 21):    number of gaze shift in trial

    %% DEBUG
    minDist_leaveReentry = 50; % Minimum distance between leaving an AOI and re-entering the same AOI
    minDur_aoiVisit      = 50; % Minimum duration of stay in AOI after entering it via a gaze shift
    minDur_sacc          = 3;  % Minimum duration of a saccade
    surround_aoi         = 20; % Area around AOI which we check for missing datasamples


    %% Find all saccade on- and offsets in gaze trace
    % We add a zero at the beginning/the end of a trace, so we can
    % determine an onset/offset for saccades that begun before stimulus
    % onset or ended after response. For those saccades, the onset/offset
    % is set to the first/last sample, which allows us to get a trace for
    % those events and to determine if any AOI visit falls within those
    % traces
    sacc_onsets  = find(diff(bitget([zeros(1); gazeTrace(:, 4)], 1)) == 1);
    sacc_offsets = find(diff(bitget([gazeTrace(:, 4); zeros(1)], 1)) == -1);
    sacc_onOff   = [];
    if ~isempty(sacc_onsets) && ~isempty(sacc_offsets)

        % For each detected onset, determine an offset
        sacc_onOff(:, 1) = sacc_onsets;
        sacc_onOff(:, 2) = NaN;
        no_saccInTrial   = size(sacc_onOff, 1);
        idx_excld        = [];
        for s = 1:no_saccInTrial % Saccades in trial

            bitToCheck = 1;
            sacc_onOff(s, 2) = ...
                infSampling_eventOffset(sacc_onOff(s, 1), sacc_offsets, gazeTrace(:, 4), bitToCheck);
            clear bitToCheck

            % Check for very short saccades
            % Those sometimes occur after a "pure" blink, which was not
            % led/followed by a saccade; we treat those very short saccades
            % as false-positives
            if sacc_onOff(s, 2) - sacc_onOff(s, 1) < minDur_sacc

                idx_excld = [idx_excld; s];

            end

        end
        sacc_onOff(idx_excld, :) = [];
        clear s no_saccInTrial idx_excld

        % Add saccade on- and offset timestamps/coordinates, saccade
        % duration, saccade latency and a flag, marking an entry in the
        % matrix as a saccade
        sacc_onOff(:, end+1:end+3) = gazeTrace(sacc_onOff(:, 1), 1:3);    % Onset timestamp and x-/y-coordinates
        sacc_onOff(:, end+1:end+3) = gazeTrace(sacc_onOff(:, 2), 1:3);    % Offset timestamp and x-/y-coordinates
        sacc_onOff(:, end+1)       = sacc_onOff(:, 2) - sacc_onOff(:, 1); % Saccade duration    
        sacc_onOff(:, end+1)       = sacc_onOff(:, 3) - ts_stimOnset;     % Saccade latency
        sacc_onOff(:, end+1)       = 1;                                   % Flag, marking a saccade

    end
    clear sacc_onsets sacc_offsets
    

    %% Find all blink on- and offsets in gaze trace
    % Blinks were detected in one of two ways: they are either lead on and
    % followed by a saccade, in which case we already have detected them as
    % a saccade in the previous section. Alternatively, they can occur as
    % "pure" blinks, in which case they are not lead on/followed by a
    % saccade. If a blink was led on by a saccade, we adjust the blink
    % on-/offset, so it corresponds to the saccade's on-/offset, and we
    % remove the false saccade from the saccade matrix. As with saccades,
    % we add zeros at the start/end of the traces
    blink_onsets  = find(diff(bitget([zeros(1); gazeTrace(:, 4)], 2)) == 1);
    blink_offsets = find(diff(bitget([gazeTrace(:, 4); zeros(1)], 2)) == -1);
    blink_onOff   = [];
    if ~isempty(blink_onsets)

        % For each detected onset, determine an offset
        blink_onOff(:, 1) = blink_onsets;
        blink_onOff(:, 2) = NaN;
        no_blinkInTrial   = size(blink_onOff, 1);
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
        for b = 1:no_blinkInTrial % Blink in trial

            no_saccInTrial = size(sacc_onOff, 1);
            blink_trace    = (blink_onOff(b, 1):blink_onOff(b, 2))';
            saccBlink_idx  = [];
            for s = 1:no_saccInTrial % Saccade in trial

                sacc_trace = (sacc_onOff(s, 1):sacc_onOff(s, 2))';
                if any(ismember(sacc_trace, blink_trace))

                    blink_onOff(b, :) = sacc_onOff(s, 1:2);
                    saccBlink_idx     = [saccBlink_idx; s];

                end
                clear sacc_trace

            end
            sacc_onOff(saccBlink_idx, :) = [];
            clear blink_trace saccBlink_idx

        end
        clear b s no_blinkInTrial no_saccInTrial

        % Add flag, marking an entry in the matrix as a blink
        blink_onOff(:, end+1:end+3) = gazeTrace(blink_onOff(:, 1), 1:3);
        blink_onOff(:, end+1:end+3) = gazeTrace(blink_onOff(:, 2), 1:3);
        blink_onOff(:, end+1)       = blink_onOff(:, 2) - blink_onOff(:, 1);
        blink_onOff(:, end+1)       = NaN;
        blink_onOff(:, end+1)       = 2;

    end
    clear blink_onsets blink_offsets    


    %% Construct gaze shift matrix
    % Put all detected gaze shifts in one matrix and sort by their respective
    % onset time
    events_gs = sortrows([sacc_onOff; blink_onOff], 1);
    clear sacc_onOff blink_onOff


    %% Determine fixated AOIs
    % For each stimulus, determine if the gaze was, at any point in time,
    % within the AOI around the stimulus, how long the gaze stayed within
    % the AOI and which gaze shift led to the AOI being entered
    stay_aoi_all = [];
    maxNo_stim   = size(stimLoc_x, 2);
    unitCircle   = 0:0.01:2*pi;
    for s = 1:maxNo_stim % Stimulus

        % Check if gaze was, at any point in time, within the AOI around
        % the current stimulus "s"
        currStim_x  = stimLoc_x(s);
        currStim_y  = stimLoc_y(s);
        coord_aoi_x = cos(unitCircle) .* aoi_radius + currStim_x;
        coord_aoi_y = sin(unitCircle) .* aoi_radius + currStim_y;

        li_inAOI  = inpolygon(gazeTrace(:, 2), gazeTrace(:, 3), coord_aoi_x, coord_aoi_y);
        clear currStim_x currStim_y coord_aoi_x coord_aoi_y

        % If the AOI around the current stimulus "s" was ever entered,
        % determine when it was entered and when it was left again
        aoi_enter      = find(diff([zeros(1); li_inAOI]) == 1);
        aoi_leave      = find(diff([li_inAOI; zeros(1)]) == -1);
        aoi_enterLeave = [];
        if ~isempty(aoi_enter)

            aoi_enterLeave(:, 1) = aoi_enter;
            aoi_enterLeave(:, 2) = NaN;
            no_aoiEntries        = size(aoi_enterLeave, 1);
            idx_excld            = [];
            for e = 1:no_aoiEntries % AOI entries

                % For each detected AOI entry, determine an AOI leave
                bitToCheck = 1;
                aoi_enterLeave(e, 2) = ...
                    infSampling_eventOffset(aoi_enterLeave(e, 1), aoi_leave, uint8(li_inAOI), bitToCheck);
                clear bitToCheck

                % Check for dataloss during the AOI visit
                if aoi_enterLeave(e, 2)+surround_aoi > size(gazeTrace, 1)

                    trace_aoiSurround = aoi_enterLeave(e, 1)-surround_aoi:size(gazeTrace, 1);

                elseif aoi_enterLeave(e, 1)-surround_aoi < 1

                    trace_aoiSurround = 1:aoi_enterLeave(e, 2)+surround_aoi;

                else

                    trace_aoiSurround = aoi_enterLeave(e, 1)-surround_aoi:aoi_enterLeave(e, 2)+surround_aoi;

                end

                dataLoss = diff(gazeTrace(trace_aoiSurround, 1));
                if any(dataLoss ~= 1)

                    idx_excld = [idx_excld; e];

                end
                clear trace_aoiSurround dataLoss

            end
            aoi_enterLeave(idx_excld, :) = [];
            clear e no_aoiEntries idx_excld

        end
        clear aoi_enter aoi_leave

        % On some rare occasions, gaze shifts ended close to the edge of an
        % AOI and keep drifting in and out of the AOI. Instead of treating
        % those cases as multiple visits of the same AOI, we want to treat
        % them as a singular, extended visit. If one AOI was entered more
        % than once, check if there is a reasonable long time between
        % successive leaves an re-entries. If not, merge successive entries
        no_aoiEntries = size(aoi_enterLeave, 1);
        if no_aoiEntries > 1

            aoi_enterLeave = infSampling_timeBetweenVisits(aoi_enterLeave, minDist_leaveReentry);

        end

        % On some rare occasions, gaze shifts only passed an AOI midfligth,
        % while being on the way to another AOI. In those cases, the datapoints
        % of the AOI visit are fully located within the gaze-shift-trace.
        % We don't count those cases as AOI visits and exclude them
        if ~isempty(aoi_enterLeave)

            % Check if the visit in an AOI is fully located within a
            % gaze-shift-trace. If this is the case, the gaze-shift-bit
            % (blink/saccade) is turned on for each datapoint of the AOI
            % visit
            no_aoiEntries = size(aoi_enterLeave, 1);
            idx_passing   = [];
            for el = 1:no_aoiEntries % AOI visit

                idx_stay_start = aoi_enterLeave(el, 1);
                idx_stay_stop  = aoi_enterLeave(el, 2);
                stay_trace     = gazeTrace(idx_stay_start:idx_stay_stop, 1);

                no_gs = size(events_gs, 1);
                for gs = 1:no_gs % Gaze shift

                    idx_gs_start = events_gs(gs, 1);
                    idx_gs_end   = events_gs(gs, 2);
                    gs_trace     = gazeTrace(idx_gs_start:idx_gs_end, 1);
                    if all(ismember(stay_trace, gs_trace))

                        idx_passing = [idx_passing; el];

                    end
                    clear idx_gs_start idx_gs_end gs_trace

                end
                clear idx_stay_start idx_stay_stop stay_trace no_gs gs

            end
            aoi_enterLeave(idx_passing, :) = [];
            clear el idx_passing no_aoiEntries

        end

        % Check which gaze shift led to the AOI being entered/left
        no_aoiEntries   = size(aoi_enterLeave, 1);
        stay_aoi_single = [];
        for el = 1:no_aoiEntries % Number of AOI entries

            % Determine which gaze shift led to the AOI being entered.
            % In some rare cases, a gaze shift did not land within the AOI,
            % but very close to its edge and the eye, subsequently, drifted
            % into the AOI: in those cases, we adjust the entry time time
            % of the AOI as not being the offset of the corresponding gaze
            % shift, but the actual time of entry into the AOI.
            % Additionally, participants sometimes have their eye closed
            % during stimulus onset, make a saccade shortly before stimulus
            % onset or look somewhere other than the fixation cross before
            % stimulus onset; in those cases, the gaze might end up in an
            % AOI very shortly after stimulus onset, without any gaze shift
            % occuring in the cirtical interval
            idx_event_gs_enter = find(events_gs(:, 1) < aoi_enterLeave(el, 1) & ...
                                      events_gs(:, 2) >= aoi_enterLeave(el, 1), ...
                                      1, 'last');            
            if isempty(idx_event_gs_enter)                                           % AOI entered via drift

                event_gs_enter(1)   = NaN;
                event_gs_enter(2)   = aoi_enterLeave(el, 1);
                event_gs_enter(3:5) = gazeTrace(aoi_enterLeave(el, 1), 1:3);
                event_gs_enter(6:7) = NaN;
                event_gs_enter(8)   = 3;

            else                                                                     % AOI entered via gaze shift

                event_gs_enter = events_gs(idx_event_gs_enter, [1:2 6:11]);

            end
            clear idx_event_gs_enter

            % Determine which gaze shift led to the AOI being left.
            % Similar to gaze shifts that entered an AOI, gaze shifts that
            % left the AOI could be missdetected and the eye actually
            % drifted out of an AOI and did not leave it via a gaze shift
            idx_event_gs_leave = find(events_gs(:, 1) <= aoi_enterLeave(el, 2) & ...
                                      events_gs(:, 2) > aoi_enterLeave(el, 2) & ...
                                      events_gs(:, 1) > event_gs_enter(2), ...
                                      1, 'last');
            if isempty(idx_event_gs_leave)                                             % AOI left via drift

                if aoi_enterLeave(el, 2) - event_gs_enter(2) > minDur_aoiVisit

%                     keyboard

                end
                event_gs_leave(1)   = NaN;
                event_gs_leave(2)   = aoi_enterLeave(el, 2);
                event_gs_leave(3:5) = gazeTrace(aoi_enterLeave(el, 2), 1:3);
                event_gs_leave(6:7) = NaN;
                event_gs_leave(8)   = 3;

            else                                                                       % AOI left via gaze shift

                event_gs_leave = events_gs(idx_event_gs_leave, [1:2 6:11]);

            end
            clear idx_event_gs_leave

            % Check how long gaze stayed in the AOI and exclude cases in
            % which the stay was shorter than the minimum stay duration.
            % There are multiple ways in which an AOI can be entered/left:
            % to determine how long gaze stayed within an AOI, we take the
            % difference in time between gaze shift offset/entry into the
            % AOI via drift as well as gaze shift onset (when AOI was left
            % via a gaze shift) or leave of AOI via drift (if AOI was left
            % by a drift)
            if isnan(event_gs_leave(1)) && ...
               aoi_enterLeave(el, 2) - event_gs_enter(2) <= minDur_aoiVisit

                event_gs_enter(:) = NaN;
                event_gs_leave    = NaN(size(event_gs_enter));


            elseif ~isnan(event_gs_leave(1)) && ...
                   event_gs_leave(1) - event_gs_enter(2) <= minDur_aoiVisit

                event_gs_enter(:) = NaN;
                event_gs_leave    = NaN(size(event_gs_enter));

            end

            % Check if the AOI was entered/left via a blink
%             entere/left via a blink
%             if stay_aoi_single(el, 9) == 2
% 
% 
% 
%             elseif stay_aoi_single(el, 17) == 2
% 
% 
% 
%             end

            % Create output
            stay_aoi_single = [stay_aoi_single; ...
                               s aoi_enterLeave(el, :) event_gs_enter event_gs_leave (event_gs_leave(3) - event_gs_enter(3))];
            clear event_gs_enter event_gs_leave

        end
        clear el no_aoiEntries aoi_enterLeave li_inAOI

        % Save for output
        % All-NaN rows are AOI visits shorter than the minimum duration
        li_notExcluded  = all(isnan(stay_aoi_single(:, 4:end)), 2);
        stay_aoi_single = stay_aoi_single(~li_notExcluded, :);

        stay_aoi_all = [stay_aoi_all; stay_aoi_single];
        clear stay_aoi_single li_notExcluded

    end
    clear s maxNo_stim coord_unitCircle unitCircle


    %% Check if any gaze shift was targeted to screen background instead of any AOI
    % We do this by, first, checking which of the detected gaze shifts led
    % to an AOI being entered and, second, to check which of the remaining
    % gaze shifts happened during a stay within an AOI. Each gaze shift
    % that does not fall in any of those categories, is classified as a
    % gaze shift that was targetet to the background (this also includes gaze
    % shifts which were excluded, because the gaze did not stay long enough
    % in an AOI, after the AOI was entered)
    if ~isempty(stay_aoi_all)

        li_gs_aoiEnter   = all(~ismember(events_gs(:, 1:2), stay_aoi_all(:, 4:5)), 2);
        events_gs_subset = events_gs(li_gs_aoiEnter, :);
        clear li_gs_aoiEnter

    else % Sometimes, participants do not fixate a single AOI in a trial

        events_gs_subset = events_gs;

    end

    no_visits = size(stay_aoi_all, 1);
    no_gs     = size(events_gs_subset, 1);
    li_gs_bg  = logical(zeros(no_gs, 1));
    for gs = 1:no_gs % Gaze shifts

        % Check if a gaze shift was executed during a visit of an AOI
        trace_gs = (events_gs_subset(gs, 1):events_gs_subset(gs, 2))';
        li_noVis = zeros(no_visits, 1);
        for vis = 1:no_visits % AOI visits

            trace_aoiVisit = (stay_aoi_all(vis, 2):stay_aoi_all(vis, 3))';
            if all(ismember(trace_gs, trace_aoiVisit))

                li_noVis(vis) = 1;

            end
            clear trace_aoiVisit

        end
        clear trace_gs

        % If the gaze shift happened outside of any AOI visit, it is
        % classified as a gaze shift that target the screen background
        if all(li_noVis == 0)

            li_gs_bg(gs) = 1;

        end
        clear li_noVis

    end
    clear no_visits no_gs

    % Add gaze shifts to background to output
    events_gs_subset = events_gs_subset(li_gs_bg, :);
    if ~isempty(events_gs_subset)

        no_gs_bg     = size(events_gs_subset, 1); 
        stay_aoi_all = [stay_aoi_all; ...
                        zeros(no_gs_bg, 1)+99999 NaN(no_gs_bg, 2) events_gs_subset(:, [1:2 6:11]) NaN(no_gs_bg, 9)];

    end
    clear events_gs_subset no_gs_bg


    %% Sort fixated AOIs by time of entry and determine position of gaze shift in trial
    stay_aoi_all           = sortrows(stay_aoi_all, 4);
    stay_aoi_all(:, end+1) = 1:size(stay_aoi_all, 1);


    %% Plot position all stimuli and saccade endpoints
    if debug_plot_allStim

        % Compose vector with area, covered by screen, as well as locations
        % of stimuli in a trial
        x_screen = [-24.5129 -24.5129 24.5129   24.5129  -24.5129]; % Screen area
        y_screen = [-13.7834  13.7834 13.7834  -13.78341 -13.7834];
        x_gs_off = events_gs(:, 7);                                 % Gaze shift offsets
        y_gs_off = events_gs(:, 8);

        unitCircle = 0:0.01:2*pi;
        no_stim    = numel(stimLoc_x);
        for s = 1:no_stim

            currStim_x  = stimLoc_x(s);
            currStim_y  = stimLoc_y(s);
            coord_aoi_x = cos(unitCircle) .* aoi_radius + currStim_x;
            coord_aoi_y = sin(unitCircle) .* aoi_radius + currStim_y;
            clear currStim_x currStim_y

            x_screen = [x_screen NaN coord_aoi_x];
            y_screen = [y_screen NaN coord_aoi_y];
            clear coord_aoi_x coord_aoi_y

        end
        clear s unitCircle no_stim

        % Check which gaze shift offset ended somewhere on the screen,
        % outside any AOI.
        % CAUTION: "inpolygons" ASSUMES THAT HOLES IN THE BACKGROUND ARE
        % ARRANGED CCW (which is the case here by default). IF WE WOULD JUST
        % PUT IN THE COORDINATES OF THE AOIs, WITHOUT FIRST REARRANGING THEM
        % PROPERLY, "inpolygons" WOULD FAIL TO DETERMINE THAT ANY OFFSET FELL
        % WITHIN ANY AOI, EVEN IF an OFFSET WAS LOCATED WITHIN AN AOI
        % (BECAUSE, WITHOUT REARRANGING THEM CW, THE AOIs ARE TREATED AS HOLES).
        % IF WE ONLY PUT IN THE COORDINATES OF THE AOIs, WE HAVE TO FIRST ARRANGE
        % THEM CW WITH "poly2cw"
        [li_in, idx_in] = inpolygons(x_gs_off, y_gs_off, x_screen, y_screen);
        idx_in          = cell2mat(idx_in);

        % Create plot
        close all
        [f, v] = poly2fv(x_screen, y_screen);                     % Plot screen area and stimuli locations
        patch('Faces',     f, ...
              'Vertices',  v, ...
              'FaceColor', [.9 .9 .9], ...
              'EdgeColor', 'none');
        daspect([1 1 1]);

        hold on
        plot(x_gs_off(li_in),  y_gs_off(li_in), 'r.', ...         % Plot endpoints inside/outside any AOI
             x_gs_off(~li_in), y_gs_off(~li_in), 'b.');
        plot(x_gs_off(idx_in == 1), y_gs_off(idx_in == 1), 'go'); % Plot corona around endpoints outside any AOI
        hold off
        close all

    end

end