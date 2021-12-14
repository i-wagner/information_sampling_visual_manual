function [time_fix, time_resp, time_search_mean, gazeShifts_nonConsec, diff_landBg_resp] = ...
            infSampling_getDwellTime(gazeShifts, time_stimOn, time_response)

    % Removes consecutive saccades and calculates dwell time within each
    % fixated AOI
    % Input
    % gazeShifts:           matrix with all gaze shifts, detected in a trial.
    %                       Has to contain the following information in this
    %                       exact order:
    %                       (:, 1:2):     datasamples of gaze shift
    %                       (:, 3):       flag for to-be-excluded gaze shift
    %                                     1 == exclude, 0 == keep
    %                       (:, 4:6):     timestamps and coordiantes of gaze
    %                                     shift onsets
    %                       (:, 7:9):     timestamps and coordinates of gaze
    %                                     shift offsets
    %                       (:, 10):      gaze shift duration
    %                       (:, 11):      gaze shift latency
    %                       (:, 12):      gaze shift type
    %                                     1 == saccade, 2 == blink
    %                       (:, [13 15]): mean gaze position after current
    %                                     and before next gaze shift
    %                       (:, [14 16]): standard deviation of mean gaze
    %                                     position after current and before
    %                                     next gaze shift
    %                       (:, 17):      index of AOI, on which gaze shift landed
    % time_fixOn:           timestampt of fixation cross onset
    % time_stimOn           timestamp of stimulus onset
    % time_response:        timestamp of response
    % Output
    % time_fix:             non-search time component: time between stimulus 
    %                       onset and offset of first saccade
    % time_resp:            non-search time component: time between response
    %                       and offset of last saccade; if last saccade did
    %                       not land on either target, we cannot calculate
    %                       this and output "NaN"
    % time_search_mean:     x
    % gazeShifts_nonConsec: same as input "gazeShifts", but without
    %                       consecutive gaze shifts (i.e., gaze shifts that
    %                       where executed within the same AOI as the
    %                       previous gaze shift), but with two additional
    %                       columns with blink duration during AOI visit
    %                       and search time for each visited AOI (except
    %                       the last one in the trial)
    % diff_landBg_resp:     time between landing of last gaze shift on
    %                       background and response (only if the last gaze
    %                       shift did land on background, otherwise NaN)

    %% Remove gaze shifts, flaged for exclusion
    li_excld                = gazeShifts(:, 3) == 1;
    gazeShifts(li_excld, :) = [];
    clear li_excld


    %% Remove consecutive gaze shifts
    % Sometimes participants make gaze shifts within an AOI; we are not
    % interested in those, but only in the first gaze shift that led to the
    % AOI being entered and the next gaze shift that left the AOI
    li_nonConsec         = diff([zeros(1); gazeShifts(:, 17)]) ~= 0;
    gazeShifts_nonConsec = gazeShifts(li_nonConsec, :);
    clear li_nonConsec


    %% Check if last gaze shift in trial went to screen center
    % If the last gaze shift in a trial landed on the background, we
    % exclude it. Since those gaze shifts could go back to screen center,
    % before the answer is given, we can check if the last gaze shift in a
    % trial laned in an AOI around the fixation cross (here, we can either
    % use the the limits of the fixation check for the AOI radius or we can
    % use the same AOI radius we use for the other AOIs)
    diff_landBg_resp = NaN(1, 2);
    idx_lastGsBg     = find(gazeShifts_nonConsec(:, 18) == 666, 1, 'last');
    no_gs            = size(gazeShifts_nonConsec, 1);
    if idx_lastGsBg == no_gs                  % Last gaze shift in trial to background

        % Calculate time between landing on background and respoding
        if no_gs > 1 & gazeShifts_nonConsec(idx_lastGsBg-1, 18) <= 2 & ...
           ~isnan(gazeShifts_nonConsec(idx_lastGsBg, 7))

            diff_landBg_resp(1) = time_response - gazeShifts_nonConsec(idx_lastGsBg, 7);

        end

        % Check is last gaze shift landed within AOI around fixation, and
        % if it did so, remove it from further analysis
%         unitCircle  = 0:0.01:2*pi;
%         sc_aoi_x    = cos(unitCircle) .* 5 + 0;
%         sc_aoi_y    = sin(unitCircle) .* 5 + 0;
%         ep_x_lastGs = gazeShifts_nonConsec(idx_lastGs, 13);
%         ep_y_lastGs = gazeShifts_nonConsec(idx_lastGs, 15);
% 
%         li_gsInAoi = inpolygon(ep_x_lastGs, ep_y_lastGs, sc_aoi_x, sc_aoi_y);
%         if li_gsInAoi == 1
% 
%             keyboard
            gazeShifts_nonConsec(idx_lastGsBg, :) = [];
% 
%         else
% 
%             keyboard
% 
%         end
        clear unitCircle sc_aoi_x sc_aoi_y ep_x_lastGs ep_y_lastGs

    elseif gazeShifts_nonConsec(end, 18) <= 2 % Last gaze shift in trial to either target

        diff_landBg_resp(2) = time_response - gazeShifts_nonConsec(end, 7);

    end
    clear idx_lastGs no_gs


    %% Check for blinks during AOI visit
    % If a participant blinked during an AOI visit, and the gaze was in the
    % same AOI before and after the blink, we will adjust the dwell time
    % by the duration of the blink (since participants cannot see anything
    % during the blink, and thus, technially do not "dwell" during this
    % time). To determine if a participant blinked during an AOI visit, we
    % will inspect if a blink occured between visits of two subsequent,
    % unique AOIs. We add an additional index to the indices of
    % non-consecutive gaze shifts so we can, in the for-loop, also check
    % for blinks when consecutive gaze shifts occured after the last
    % non-consecutive gaze shift
    no_nonConsec  = size(gazeShifts_nonConsec, 1);
    no_gs         = size(gazeShifts, 1);
    idx_nonConsec = [find(diff([zeros(1); gazeShifts(:, 17)]) ~= 0); no_gs+1];

    gazeShifts_nonConsec(:, end+1) = zeros;
    for nc = 1:no_nonConsec % Non-consecutive gaze shifts

        % We loop through each non-consecutive gaze shift and check if a
        % blink occured in the time between offset of the current gaze shift
        % "nc" and the next non-consecutive gaze shift "nc+1"
        if diff([idx_nonConsec(nc) idx_nonConsec(nc+1)]) ~= 1

            idx_start    = idx_nonConsec(nc)+1;
            idx_end      = idx_nonConsec(nc+1)-1;
            gs_inAOI     = gazeShifts(idx_start:idx_end, :);
            li_blinkStay = gs_inAOI(:, 12) == 2;
            if any(li_blinkStay == 1)

                gazeShifts_nonConsec(nc, end) = sum(gs_inAOI(li_blinkStay, 10));

            end
            clear idx_start idx_end gs_inAOI li_blinkStay

        end

    end
    clear nc no_nonConsec no_gs idx_nonConsec idx_nonConsec


    %% Get timestamps of when gaze entered/left an AOI
    % If an AOI was entered by a gaze shift and left by a blink, we
    % calculate the search-time as the difference between the time of offset
    % of the entering gaze shift and time of onset of the leaving blink. If
    % an AOI was entered by a gaze shift and left by a gaze shift, we
    % calculate the search-time as the difference between the time of
    % offset of the entering gaze shift and time of offset of the leaving
    % gaze shift. There is no special case for when an AOI was entered by a
    % blink; here, we just take the offset of the blink as the reference
    % point. For the last gaze shift in a trial, we take the response time
    % as the leaving time
    no_nonConsec = size(gazeShifts_nonConsec, 1);
    visits_AOI   = NaN(no_nonConsec, 10);
    for gs = 1:no_nonConsec

        if gs ~= no_nonConsec && gazeShifts_nonConsec(gs+1, 12) == 1              % AOI left via saccade

            ts_leave = gazeShifts_nonConsec(gs+1, 7);

        elseif gs ~= no_nonConsec && gazeShifts_nonConsec(gs+1, 12) == 2          % AOI left via blink

            ts_leave = gazeShifts_nonConsec(gs+1, 4);

        elseif gs == no_nonConsec                                                 % Last gaze shift before response

            ts_leave = time_response;

        else

            keyboard

        end

        % Compose data of current gaze, necessary to calculate search-time
        % Time of entering gaze shift, time of leaving gaze shift, type,
        % mean gaze position, target AOI  of entering gaze shift and
        % duration of blinks during AOI visit
        visits_AOI(gs, :) = [gazeShifts_nonConsec(gs, 7) ts_leave ... 
                             gazeShifts_nonConsec(gs, 12:end)];
        clear ts_leave

    end
    clear gs no_nonConsec


    %% Calculate search and non-search time
    % Search times
    % We define search as the time between entering and leaving an AOI.
    % Search time is not calculated for gaze shifts that landed on the
    % background and for the last gaze shift in a trial. If a blink was
    % detected during an AOI visit, we subtract the blink duration from the
    % search time of the stimulus
    no_visits          = size(visits_AOI, 1);
    time_search_single = NaN(no_visits, 1);
    li_dwells          = logical([visits_AOI(1:end-1, 8) < 666; 0]);

    time_search_single(li_dwells)  = visits_AOI(li_dwells, 2) - visits_AOI(li_dwells, 1) - visits_AOI(li_dwells, 10);
    gazeShifts_nonConsec(:, end+1) = time_search_single;
    time_search_mean               = nanmean(time_search_single);
    clear li_dwells

    % Non-search time
    % We define non-search time as the sum of the time between stimulus
    % onset and first saccade offset as well as the time between last
    % saccade offset and response
    time_fix = NaN;
    if no_visits > 0

        time_fix = visits_AOI(1, 1) - time_stimOn;

    end

    % The response time can only be calculated if a participants last
    % gaze shift either landed on a target or, if it did not land on a
    % target, if gaze shifts only landed on the background after the last
    % gaze shift to a target
    time_resp    = NaN;
    idx_lastTarg = find(visits_AOI(:, 8) <= 2, 1, 'last');
    if ~isempty(idx_lastTarg) && ...
       all(visits_AOI(idx_lastTarg:end, 8) < 3 | visits_AOI(idx_lastTarg:end, 8) == 666)

        time_resp = time_response - visits_AOI(idx_lastTarg, 1) - visits_AOI(idx_lastTarg, 10);

    end
    clear idx_lastTarg

end