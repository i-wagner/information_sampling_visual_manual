close all; clear all; clc;


%% Go to folder with data
cd(fileparts(which(mfilename)));
cd ..

exp.name.root     = pwd;
exp.name.data     = strcat(exp.name.root, '/', '2_data');
exp.name.analysis = strcat(exp.name.root, '/', '3_analysis');
exp.name.plt      = strcat(exp.name.root, '/', '4_figures');

addpath(exp.name.analysis, strcat(exp.name.analysis, '/_model'));
cd(exp.name.data);


%% Experiment structure
exp.num.conds  = [2; 3]; % 2 == single-target, 3 == double-target
exp.num.subs   = (1:21)';
exp.num.subNo  = numel(exp.num.subs);
exp.num.condNo = numel(exp.num.conds);


%% Plot settings
plt.name.aggr     = strcat(exp.name.plt, '/infSampling_c_'); % Naming scheme; aggregated data
plt.color         = plotColors;                              % Default colors
plt.color.o1      = [241 163 64]./255;
plt.color.o2      = [239 209 171]./255;
plt.color.p1      = [153 142 195]./255;
plt.color.p2      = [199 187 245]./255;
plt.color.mid     = [219 198 208]./255;
plt.color.target  = {'#384B60' '#4A544A'};                   % Some additional, special colors
plt.color.distrct = {'#5C93C4' '#849D6A'};
plt.color.bg      = {'#9C2E8F' '#FF9BC7'};
plt.size.mrk_ss   = 8;                                       % Markersize when plotting single-subject data
plt.size.mrk_mean = 10;                                      % Markersize when plotting aggregated data
plt.lw.thick      = 2;                                       % Linewidth
plt.lw.thin       = 1;
plt.save          = 0;                                       % Toggle if plots should be saved to drive
show_figs         = 1;                                       % Toggle if figure should be shown during plotting
if show_figs == 1

    set(groot, 'DefaultFigureVisible', 'on')

else

    set(groot, 'DefaultFigureVisible', 'off')

end
clear show_figs


%% Miscellaneous settings
exp.avg.minSub = 10; % Minimum number of subjects required to calculate mean
exp.avg.minDp  = 10; % Minimum number of datapoints (single-subject-level) required to calculate mean

% Toggle if data should be exported for model
exp.flag.export = 0;


%% Settings of screen, on which data was recorded
screen = screenBig;


%% Stimulus settings
% We define our AOI as a circular area, with a diameter of 5deg, around the
% center of each stimulus
stim.diameter.px     = 49;                                 % Stimulus diameter (pixel)
stim.diameter.deg    = stim.diameter.px * screen.xPIX2DEG; % Stimulus diameter (deg)
stim.diameterAOI.deg = 5;                                  % AOI diameter (deg)
stim.radiusAOI.deg   = stim.diameterAOI.deg / 2;           % AOI radius (deg)

% Identifier for easy (:, 1) and hard (:, 2) targets (1, :) and distractors
% (2, :)
stim.identifier = [1 2; 3 4];


%% Define columns for log file
log.col.trialNo      = 4;     % Trial number
log.col.noTargets    = 5;     % Number of targets in trial
log.col.targetDiff   = 6;     % Target type shown in trial (only single-target experiment); 1 == easy, 2 == hard
log.col.diffLvlEasy  = 7;     % Difficulty level of easy target
log.col.diffLvlHard  = 8;     % Difficulty level of hard target
log.col.gapPosEasy   = 9;     % Gap location on easy target; 1 == bottom, 2 == top, 3 == left, 4 == right
log.col.gapPosHard   = 10;    % Gap location on hard target
log.col.gapPosReport = 11;    % Reported gap location
log.col.noDisEasy    = 12;    % Number of easy distractors in trial
log.col.noDisHard    = 13;    % Number of hard distractors in trial
log.col.noDisOverall = 14;    % Overall number of distractors in trial
log.col.stimPosX     = 15:24; % Positions on x-axis
log.col.stimPosY     = 25:34; % Positions on y_axis
log.col.cumTimer     = 35;    % Cumulative timer
log.col.hitMiss      = 36;    % Hit/miss
log.col.score        = 37;    % Number of points
log.col.fixErr       = 38;    % Flag for fixation error


%% Get data from trials
% Allocate memory
exp.trialNo           = NaN(exp.num.subNo, exp.num.condNo);  % Number of solved trials
exp.excl_trials       = cell(exp.num.subNo, exp.num.condNo); % Trials with fixation error
exp.events.stim_onOff = cell(exp.num.subNo, exp.num.condNo); % Timestamps of stimulus on- and offset
exp.cum_trialTime     = cell(exp.num.subNo, exp.num.condNo); % Cumulative time spent on a trial
sacc.gazeShifts       = cell(exp.num.subNo, exp.num.condNo); % Gaze shifts (blinks and saccades)
sacc.gazeShifts_zen   = cell(exp.num.subNo, exp.num.condNo); % Gaze shifts (blinks and saccades) for Zenodo
sacc.time.resp        = cell(exp.num.subNo, exp.num.condNo); % Response times
sacc.time.fix         = cell(exp.num.subNo, exp.num.condNo); % Fixation times
sacc.time.search      = cell(exp.num.subNo, exp.num.condNo); % Search times
sacc.time.resp_bg     = cell(exp.num.subNo, exp.num.condNo); % Time between last gaze shift on background and response
sacc.time.inspecting  = cell(exp.num.subNo, exp.num.condNo); % Time that was spent searching in a trial (counterintuitive naming, since we changed naming scheme while writing manuscript)
sacc.propGs.closest   = cell(exp.num.subNo, exp.num.condNo); % Proportion gaze shifts to closest stimulus
sacc.propGs.further   = cell(exp.num.subNo, exp.num.condNo); % Proportion gaze shifts to further away stimulus
stim.chosenTarget     = cell(exp.num.subNo, exp.num.condNo); % Chosen target
stim.choiceCorrespond = cell(exp.num.subNo, exp.num.condNo); % Corresponde chosen and last saccade target
stim.no_easyDis       = cell(exp.num.subNo, exp.num.condNo); % # easy distractors
stim.no_hardDis       = cell(exp.num.subNo, exp.num.condNo); % # difficult distractors
perf.score.final      = NaN(exp.num.subNo, exp.num.condNo);  % Score at end of condition
perf.hitMiss          = cell(exp.num.subNo, exp.num.condNo); % Hit/miss in trial
for c = 1:exp.num.condNo % Condition

    curr_cond = exp.num.conds(c);
    for s = 1:exp.num.subNo % Subject

        % Go to folder of single subject
        curr_sub    = exp.num.subs(s);
        dirName_sub = sprintf('e%dv%db1', curr_cond, curr_sub);
        if isfolder(dirName_sub)

            cd(dirName_sub);

        else

            disp(['Skipping missing participant ', num2str(curr_sub)])
            continue

        end
        clear dirName_sub

        % Load .log file of single subject and extract relevant data from it
        fileName_log = sprintf('e%dv%db1.log', curr_cond, curr_sub);
        log.file     = load(fileName_log);
        clear fileName_log
%         test=NaN(2, 8);
%         for d = 0:8
%             test(1, d+1) = sum(log.file(:, 6) == 1 & log.file(:, 12) == d);
%             test(2, d+1) = sum(log.file(:, 6) == 2 & log.file(:, 13) == d);
%         end
%         clc; test
%         test=NaN(1, 8);
%         for d = 0:8
%             test(d+1) = sum(log.file(:, 12) == d);
%         end
%         clc; test

        exp.trialNo(curr_sub, c)      = max(log.file(:, log.col.trialNo));      % # completed trials
        exp.excl_trials{curr_sub, c}  = find(log.file(:, log.col.fixErr) == 1); % Trials with fixation error
        perf.score.final(curr_sub, c) = log.file(end, log.col.score);           % Score of subject at end of condition
        perf.hitMiss{curr_sub, c}     = log.file(:, log.col.hitMiss);           % Hit/miss in trial
        stim.no_easyDis{curr_sub, c}  = log.file(:, log.col.noDisEasy);         % # easy distractors in trial
        stim.no_hardDis{curr_sub, c}  = log.file(:, log.col.noDisHard);         % # difficult distractors in trial

        % Iterate through trials and get gaze shifts, fixated AOIs and
        % search as well as non-search times
        no_trials_singleSub = exp.trialNo(curr_sub, c);    % Number of trials

        trial.events.stim_onOff  = NaN(no_trials_singleSub, 2); % Timestamps of stimulus on- and offset
        time_fix                 = NaN(no_trials_singleSub, 1); % Fixation times
        time_resp                = NaN(no_trials_singleSub, 1); % Response times
        time_respBg              = NaN(no_trials_singleSub, 2); % Time between last gaze shift on background and response
        time_search              = NaN(no_trials_singleSub, 1); % Search times
        time_trial               = NaN(no_trials_singleSub, 1); % Time spent per trial
        choice_target            = NaN(no_trials_singleSub, 1); % Chosen target
        choice_congruence        = NaN(no_trials_singleSub, 1); % Corresponde responded and last fixated target
        prop_gsClosest           = NaN(no_trials_singleSub, 1); % Proportion gaze shifts to closest AOI
        prop_gsFurther           = NaN(no_trials_singleSub, 1); % Proportion gaze shifts to more distant AOI
        inspectedElements_no     = NaN(no_trials_singleSub, 3); % # unique stimuli a subjects inspected
        gazeShifts_allTrials_zen = [];                          % Gaze shift matrix for Zenodo
        gazeShifts_allTrials     = [];                          % Gaze shift matrix for analysis
        for t = 1:no_trials_singleSub % Trial

            % Get gaze trace in trial
            [trial.gazeTrace, flag_dataLoss] = loadDat(t, screen.x_pix, screen.y_pix);
            exp.excl_trials{curr_sub, c}     = [exp.excl_trials{curr_sub, c}; flag_dataLoss];
            clear flag_dataLoss

            % Convert pixel to degrees visual angle and flip y-axis coordinates
            % (so that positive values correspond to stimulus locations in
            % the upper screen half)
            flip_yAxis = 1;
            [trial.gazeTrace(:, 2), trial.gazeTrace(:, 3)] = ...
                convertPix2Deg(trial.gazeTrace(:, 2), trial.gazeTrace(:, 3), ...
                               [screen.x_center screen.y_center], ...
                               [screen.xPIX2DEG screen.yPIX2DEG], ...
                               flip_yAxis);
            clear flip_yAxis

            % Get events in trial
            % We expect five events to happen in a trial:
            % trial begin, start recording, fixation onset, onset of stimuli
            % and offset of stimuli [i.e., response])
            trial.events.all = find(bitget(trial.gazeTrace(:, 4), 3));
            if numel(trial.events.all) ~= 5

                % For one participant in the single-target condition, we
                % miss a chunk of data right a trial start, which results
                % in an event being missing. To stop the analysis from
                % crashing, we add a placeholder so that the event vector
                % has the appropriate length
                trial.events.all = [trial.events.all(1:2); NaN; trial.events.all(3:end)];

            end
            trial.events.stim_onOff(t, 1) = trial.gazeTrace(trial.events.all(4), 1); % Timestamp stimulus onset
            trial.events.stim_onOff(t, 2) = trial.gazeTrace(trial.events.all(5), 1); % Timestamp stimulus offset

            % Get how much time passed between stimulus on- and offset
            time_trial(t) = trial.events.all(5) - trial.events.all(4);

            % Offline check for fixation errors (just to be sure)
            fixPos_stimOn  = trial.gazeTrace(trial.events.all(4)-20:trial.events.all(4)+80, 2:3);
            fixPos_deviate = sum(abs(fixPos_stimOn(:)) > 2);
            if fixPos_deviate > 0 && ~ismember(t, exp.excl_trials{curr_sub, c})

                keyboard
                exp.excl_trials{curr_sub, c} = [exp.excl_trials{curr_sub, c}; t];

            end
            clear fixPos_stimOn fixPos_deviate

            % The stimulus locations, extracted from the .log-file, are not
            % ordered: because of this, different cells in the location
            % vector might correspond to different stimuli types, depending
            % on the number of easy/hard distractors in a trial. To make
            % our life easier, we want to order them so that each position
            % in the location matrix is directly linked to one type of
            % stimulus (easy/hard target/distractor)
            % CAUTION: SINCE WE FLIPPED THE Y-COORDINATES OF THE GAZE TRACE
            % WE ALSO FLIP THE Y-COORDINATES OF THE STIMULUS LOCATIONS
            inp_x_loc   = log.file(t, log.col.stimPosX);       % Stimulus locations
            inp_y_loc   = log.file(t, log.col.stimPosY) .* -1;
            inp_no_ed   = log.file(t, log.col.noDisEasy);      % Set-size easy distractors
            inp_no_dd   = log.file(t, log.col.noDisHard);      % Set-size hard distractors
            inp_no_targ = curr_cond-1;                         % Number targets in condition
            if curr_cond == 2

                inp_shownTarg = log.file(t, log.col.targetDiff); % Target shown in trial

            elseif curr_cond == 3 % In the double-target condition, both target were always shown

                inp_shownTarg = NaN;
                
            end

            stim_locations = ...                  
                infSampling_getStimLoc(inp_x_loc, inp_y_loc, ...
                                       inp_no_ed, inp_no_dd, ...
                                       inp_no_targ, inp_shownTarg);
            clear inp_x_loc inp_y_loc inp_no_ed inp_no_dd inp_no_targ inp_shownTarg

            % Get gaze shifts in trial
            % Gaze shifts are all saccades and blinks, detected in a trial
            inp_gazeTrace   = trial.gazeTrace(trial.events.all(4):trial.events.all(5), :); % Gaze trace between stimulus on- and offset
            inp_ts_stimOn   = trial.events.stim_onOff(t, 1);                               % Timestampe stimulus onset
            inp_minDur_sacc = 5;                                                           % Minimum duration of gaze shifts [ms]; everything beneath is flagged
            inp_screen_x    = (screen.x_pix - screen.x_center) * screen.xPIX2DEG;          % Most extrem gaze position possible, given the display size
            inp_screen_y    = (screen.y_pix - screen.y_center) * screen.yPIX2DEG;

            gazeShifts_singleTrial = ...
                infSampling_getGazeShifts(inp_gazeTrace, inp_ts_stimOn, ...
                                          inp_minDur_sacc, inp_screen_x, inp_screen_y);
            clear inp_gazeTrace inp_ts_stimOn inp_minDur_sacc inp_screen_x inp_screen_y

            % Get fixated AOIs
            % If a gaze shift landed on either one of the stimuli, we
            % output the index of this stimulus in the location matrix; if
            % a gaze shift landed on the background (i.e., the area not
            % covered by any AOI) we output an unambigious flag.
            % To determine the fixated AOI, we use the mean gaze position
            % between end of the current gaze shift and the begin of the
            % next gaze shift. We do this, because, sometimes, a gaze shift
            % might initially land in an AOI/land close to the edge of an AOI,
            % but then drift out/drift into the AOI. By using the mean gaze
            % position after each gaze shift we can circumvent those kind
            % of fluctuations and get more reliable estimate for if an AOI
            % was fixated or not
            inp_endpoints_x = gazeShifts_singleTrial(:, 13); % Mean gaze position after gaze shift
            inp_endpoints_y = gazeShifts_singleTrial(:, 15);
            inp_stimLoc_x   = stim_locations(:, :, 1);       % Stimulus locations
            inp_stimLoc_y   = stim_locations(:, :, 2);
            inp_aoi_radius  = stim.radiusAOI.deg;            % Desired AOI size
            inp_debug_plot  = 0;                             % Plot stimulus locations and gaze shift endpoints

            gazeShifts_singleTrial(:, end+1) = ...
                infSampling_getFixatedAOI(inp_endpoints_x, inp_endpoints_y, ...
                                          inp_stimLoc_x, inp_stimLoc_y, ...
                                          inp_aoi_radius, inp_debug_plot);
            clear inp_endpoints_x inp_endpoints_y inp_stimLoc_x inp_stimLoc_y inp_aoi_radius inp_debug_plot

            % Currently, each individual distractor has an unique identifier,
            % which corresponds to the distractors location in the position
            % matrix; to make things easier, we assign easy/hard
            % distractors an unambigous identifier each. For targets and
            % the background the identifier stays the same as before
            li_de = gazeShifts_singleTrial(:, end) > 2 & gazeShifts_singleTrial(:, end) <= 10;
            li_dd = gazeShifts_singleTrial(:, end) > 10 & gazeShifts_singleTrial(:, end) <= 18;
            li_d  = sum([li_de li_dd ], 2);

            gazeShifts_singleTrial(:, end+1)   = zeros;
            gazeShifts_singleTrial(li_de, end) = stim.identifier(2, 1);                % Easy distractor
            gazeShifts_singleTrial(li_dd, end) = stim.identifier(2, 2);                % Hard distractor
            gazeShifts_singleTrial(~li_d, end) = gazeShifts_singleTrial(~li_d, end-1); % Other
            clear li_de li_dd li_d

            % Get unique AOI fixations as well as search and non-search times
            % Sometimes participants land in an AOI and make corrective
            % gaze shifts within this AOI, without actually leaving the
            % AOI; we call those consecutive gaze shifts and we are not
            % interested in them
            % search times:    time between entering and leaving an AOI
            %                  entered/left via saccade:          time between entering and leaving saccade offset
            %                  entere via saccade/left via blink: time between entering saccade offset and blink onset
            %                  blink during visit:                duration of blink subtracted from search time
            % non-search time: sum of response time (between offset of last
            %                  gaze shift and response) and time between
            %                  stimulus onset and offset of first gaze
            %                  shift. Response time can only be calculated
            %                  if the last gaze shift landed in a target
            inp_gazeShifts = gazeShifts_singleTrial;
            inp_stimOn     = trial.events.stim_onOff(t, 1);
            inp_stimOff    = trial.events.stim_onOff(t, 2);

            [time_fix(t), time_resp(t), time_search(t), gazeShifts_noConsec_singleTrial, time_respBg(t, :)] = ...
                infSampling_getDwellTime(inp_gazeShifts, inp_stimOn, inp_stimOff);
            clear inp_gazeShifts inp_stimOn inp_stimOff

            % Check if gaze shifts went to closest stimulus 
            inp_gsOn_x    = gazeShifts_noConsec_singleTrial(:, 5);  % Onset position of gaze shifts
            inp_gsOn_y    = gazeShifts_noConsec_singleTrial(:, 6); 
            inp_targAoi   = gazeShifts_noConsec_singleTrial(:, 17); % Index of gaze shift target
            inp_flagBg    = 666;                                    % Flag, marking background as gaze shift target
            inp_stimLoc_x = stim_locations(:, :, 1);                % Stimulus locations
            inp_stimLoc_y = stim_locations(:, :, 2);

            [gazeShifts_noConsec_singleTrial(:, end+1), prop_gsClosest(t), prop_gsFurther(t)] = ...
                infSampling_distStim(inp_gsOn_x, inp_gsOn_y, inp_targAoi, ...
                                     inp_stimLoc_x, inp_stimLoc_y, inp_flagBg);

            % Check distance between gaze while fixating and closest stimulus
            inp_gsOn_x  = gazeShifts_noConsec_singleTrial(:, 13); % Onset position of gaze shifts
            inp_gsOn_y  = gazeShifts_noConsec_singleTrial(:, 15);
            inp_targAoi = NaN(size(inp_gsOn_x, 1), 1);            % Do not correct for currently fixated AOI

            [~, ~, ~, dist] = ...
                infSampling_distStim(inp_gsOn_x, inp_gsOn_y, inp_targAoi, ...
                                     inp_stimLoc_x, inp_stimLoc_y, inp_flagBg);
            clear inp_gsOn_x inp_gsOn_y inp_targAoi inp_flagBg inp_stimLoc_x inp_stimLoc_y

            % Get chosen target in trial
            % 1 == easy target, 2 == hard target
            if curr_cond == 3 % Double-target condition

                % Chosen target is the one a participant looked at last,
                % before giving a response. If the last fixated AOI was the
                % background and the second-to-last a target, this target
                % is the chosen target. If something else is fixated, no
                % chosen target can be identified
                inp_gapLoc_easy = log.file(t, log.col.gapPosEasy);
                inp_resp        = log.file(t, log.col.gapPosReport);
                inp_fixAOI      = gazeShifts_noConsec_singleTrial(:, 18);
                inp_flag_targ   = stim.identifier(1, :);
                inp_flag_dis    = stim.identifier(2, :);
                inp_flag_bg     = 666;

                [choice_target(t), ~, choice_congruence(t)] = ...
                    infSampling_getChosenTarget(inp_gapLoc_easy, inp_resp, inp_fixAOI, ...
                                                inp_flag_targ, inp_flag_dis, inp_flag_bg);
                clear inp_gapLoc_easy inp_resp inp_fixAOI inp_flag_targ inp_flag_dis inp_flag_bg

            elseif curr_cond == 2 % Single-target condition

                % Chosen target is the target shown in trial
                choice_target(t) = log.file(t, log.col.targetDiff);

            end

            % Count how many unique stimuli were fixated in a trial,
            % determine how much time was spent searching in a trial (#
            % fixated stimuli * search time in a trial) and add #
            % distractors (of the chosen set) in a trial
            % If one and the same stimulus was fixated more than once, we
            % treat this as one fixation
            inspectedElements_no(t, 1) = infSampling_getUniqueFixations(gazeShifts_noConsec_singleTrial(:, 17), ...
                                                                        [stim.identifier(1, 1) stim.identifier(1, 2)], ...
                                                                        666);
            inspectedElements_no(t, 2) = inspectedElements_no(t, 1) * time_search(t);
            if choice_target(t) == 1

                inspectedElements_no(t, 3) = log.file(t, log.col.noDisEasy);

            elseif choice_target(t) == 2

                inspectedElements_no(t, 3) = log.file(t, log.col.noDisHard);

            end

            % Create gaze shift matrix
            % Gaze shift matrix, used for further analysis
            % This one contains all non-consecutive gaze shifts in a trial
            % and some additional data (set-size easy, chosen target,
            % timelock to trial start/last gaze shift in trial and trial
            % number)
            % (:, 1:2):   indices of onset/offset of gaze shift
            % (:, 3):     bit gaze shift onset
            % (:, 4:6):   timestamp, x- and y-coordinates of gaze shift onset
            % (:, 7:9):   timestamp, x- and y-coordinates of gaze shift offset
            % (:, 10):    gaze shift duration
            % (:, 11):    gaze shift latency
            % (:, 12):    flag, marking a gaze shift as saccade ("1") or
            %             blink ("2")
            % (:, 13:14): mean and standard deviation of horizontal gaze
            %             position after gaze shift offset
            % (:, 15:16): mean and standard deviation of vertical gaze
            %             position after gaze shift offset
            % (:, 17):    index of AOI in location matrix, target by gaze
            %             shift
            % (:, 18):    flag, marking if a gaze shift went to easy target
            %             ("1"), hard target ("2"), easy distractor ("3"),
            %             hard distractor("4") or background ("666")
            % (:, 19):    time to subtract from dwell-time; corresponds to
            %             duration of blinks that happened during AOI
            %             visit
            % (:, 20):    dwell-time within AOI
            % (:, 21):    flag if gaze shift target stimulus, closest to
            %             gaze shift onset location ("1") or not ("0")
            % (:, 22):    # easy distractors
            % (:, 23):    easy ("1") or hard ("2") target chosen
            % (:, 24):    # gaze shift after trial start
            % (:, 25):    # gaze shift before last gaze shift
            % (:, 26):    trial number, from which we extracted gaze shift
            % (:, 27):    distance between mean fixation position and closest stimulus
            no_gs_ncs = size(gazeShifts_noConsec_singleTrial, 1);

            gazeShifts_allTrials = [gazeShifts_allTrials; ...
                                    gazeShifts_noConsec_singleTrial ...                    Gaze shifts in trial 
                                    zeros(no_gs_ncs, 1)+log.file(t, log.col.noDisEasy) ... Number easy distractors
                                    zeros(no_gs_ncs, 1)+choice_target(t) ...               Chosen target
                                    (1:no_gs_ncs)' ...                                     Timelock to trial start
                                    (no_gs_ncs-1:-1:0)' ...                                Timelock to last gaze shift
                                    zeros(no_gs_ncs, 1)+t, ...                             Trial number
                                    dist(:, end)];                                       % Distance
            clear no_gs_ncs gazeShifts_noConsec_singleTrial

            % Gaze shift matrix, used for export to Zenodo
            % This one contains all gaze shifts, including consecutive gaze
            % shifts and some additional data (set-size easy, chosen target, 
            % timestamps for stimulus on- and offset and trialnumber). For
            % the gaze shifts, we export timestamps and coordinates of on-
            % and offset, type of gaze shift (saccade/blink) and mean
            % and std of gaze between AOI visits
            no_gs = size(gazeShifts_singleTrial, 1);

            gazeShifts_allTrials_zen = [gazeShifts_allTrials_zen; ...
                                        gazeShifts_singleTrial(:, [4:9 12:16]) ...          Gaze shifts in trial
                                        zeros(no_gs, 1)+log.file(t, log.col.noDisEasy) ...  Number easy distractors
                                        zeros(no_gs, 1)+choice_target(t) ...                Chosen target
                                        zeros(no_gs, 2)+trial.events.stim_onOff(t, :) ...   Timestamps stimulus on-/offset
                                        zeros(no_gs, 1)+t];                               % Trial number
            clear no_gs gazeShifts_singleTrial stim_locations

        end
        clear no_trials_singleSub

        % Store data of subject
        % inspectedElements_no(t, 1, s, c) = no_unique_aoi_fixated;
        exp.events.stim_onOff{curr_sub, c} = trial.events.stim_onOff;  % Timestamps of stimulus on- and offset
        exp.cum_trialTime{curr_sub, c}     = time_trial;               % Time spent on trial
        stim.chosenTarget{curr_sub, c}     = choice_target;            % Target, chosen in trial
        stim.choiceCorrespond{curr_sub, c} = choice_congruence;        % Correspondece between last fixated and responded on target
        sacc.gazeShifts{curr_sub, c}       = gazeShifts_allTrials;     % Non-consecutive gaze shifts
        sacc.gazeShifts_zen{curr_sub, c}   = gazeShifts_allTrials_zen; % All gaze shifts (for Zenodo)
        sacc.time.resp{curr_sub, c}        = time_resp;                % Response times (time between last saccade offset and response)
        sacc.time.fix{curr_sub, c}         = time_fix;                 % Fixation times (time between first saccade offset and stimulus onset)
        sacc.time.search{curr_sub, c}      = time_search;              % Search times (for each fixated stimulus, time between entering gaze
                                                                       % shift onset and time leaving gaze shift offset)
        sacc.time.resp_bg{curr_sub, c}     = time_respBg;              % Time between last gaze shift in background and response
        sacc.propGs.closest{curr_sub, c}   = prop_gsClosest;           % Proportion gaze shifts to closest AOI
        sacc.propGs.further{curr_sub, c}   = prop_gsFurther;           % Proportion gaze shifts to closest AOI
        sacc.time.inspecting{curr_sub, c}  = inspectedElements_no;     % # inspected elements & time spent inspecting in trial
        clear t curr_sub trial gazeShifts_allTrials choice_target time_resp time_fix time_search
        clear prop_gsClosest prop_gsFurther choice_congruence time_respBg inspectedElements_no time_trial

        cd(exp.name.data);

    end
    clear s curr_cond

end
clear c log
cd(exp.name.root);


%% Export for Zenodo
clear gazeShifts_allTrials_zen


%% Exclude invalid trials and check data quality
exp.prop.val_trials = NaN(exp.num.subNo, exp.num.condNo);
for c = 1:exp.num.condNo % Condition

    for s = 1:exp.num.subNo % Subject

        curr_sub  = exp.num.subs(s);
        idx_excld = sort(unique(exp.excl_trials{curr_sub, c}));

        exp.events.stim_onOff{curr_sub, c}(idx_excld, :, :) = NaN;
        sacc.time.resp{curr_sub, c}(idx_excld, :)           = NaN;
        sacc.time.fix{curr_sub, c}(idx_excld, :)            = NaN;
        sacc.time.search{curr_sub, c}(idx_excld, :)         = NaN;
        sacc.time.resp_bg{curr_sub, c}(idx_excld, :)        = NaN;
        sacc.time.inspecting{curr_sub, c}(idx_excld, :)     = NaN;
        sacc.propGs.closest{curr_sub, c}(idx_excld, :)      = NaN;
        sacc.propGs.further{curr_sub, c}(idx_excld, :)      = NaN;
        stim.chosenTarget{curr_sub, c}(idx_excld, :)        = NaN;
        stim.choiceCorrespond{curr_sub, c}(idx_excld, :)    = NaN;
        stim.no_easyDis{curr_sub, c}(idx_excld, :)          = NaN;
        stim.no_hardDis{curr_sub, c}(idx_excld, :)          = NaN;
        perf.hitMiss{curr_sub, c}(idx_excld, :)             = NaN;

        gazeShifts     = sacc.gazeShifts{curr_sub, c};
        gazeShifts_zen = sacc.gazeShifts_zen{curr_sub, c};
        if ~isempty(gazeShifts)

            li_excld                         = ismember(gazeShifts(:, 26), idx_excld);
            li_excld_zen                     = ismember(gazeShifts_zen(:, 16), idx_excld);
            gazeShifts(li_excld, :)          = NaN;
            gazeShifts_zen(li_excld, :)      = NaN;
            sacc.gazeShifts{curr_sub, c}     = gazeShifts;
            sacc.gazeShifts_zen{curr_sub, c} = gazeShifts_zen;

        end
        clear gazeShifts gazeShifts_zen li_excld li_excld_zen

        % Calculat proportion valid trials
        exp.prop.val_trials(curr_sub, c) = 1 - numel(idx_excld) / exp.trialNo(curr_sub, c);
        clear idx_excld

    end
    clear s curr_sub

end
clear c curr_cond


%% Exclude subjects
% We exclude subjects based on the following criteria:
% -- For some reason, only participated in one out of two conditions
%    For this, we just check if the number of completed trials is missing
%    for one of the conditions; if it is, the conditions was not done
% -- Too little proportion trials in double-target condition with
%    correspondence between last fixated and responded on target
% -- Too little trials with response time?

% Calculate proportion trials in which last fixated and responded on target
% corresponded as well as proportion trials in which where able to calculate
% response time
exp.prop.resp_trials       = NaN(exp.num.subNo, exp.num.condNo);
exp.prop.correspond_trials = NaN(exp.num.subNo, exp.num.condNo-1);
for c = 1:exp.num.condNo % Condition

    curr_cond = exp.num.conds(c);
    for s = 1:exp.num.subNo % Subject

        curr_sub  = exp.num.subs(s);
        idx_excld = sort(unique(exp.excl_trials{curr_sub, c}));

        % Calculat proportion trials for which we could calculate the response time
        time_resp = sacc.time.resp{curr_sub, c};

        exp.prop.resp_trials(curr_sub, c) = sum(~isnan(time_resp)) / (exp.trialNo(curr_sub, c) - numel(idx_excld));
        clear time_resp

        % Calculate proportion trials in which the last fixated and the
        % responded on target corresponded
        if curr_cond == 3 % Only doube-target condition

            no_valid      = exp.trialNo(curr_sub, c) - numel(idx_excld);  % # valid trials
            no_correspond = sum(stim.choiceCorrespond{curr_sub, c} == 1); % # trials with correspondence

            exp.prop.correspond_trials(curr_sub) = no_correspond / no_valid;
            clear no_valid no_correspond

        end
        clear idx_excld

    end
    clear s curr_sub

    % Plot proportion trials where respond and last fixated target
    % correspond and do not correspond
    if curr_cond == 3 % Only doube-target condition

        corr_dat = [exp.prop.correspond_trials ...
                    1-exp.prop.correspond_trials];
        lat_dat = vertcat(sacc.time.resp_bg{:, c});

        fig.h = figure;

        infSampling_plt_propCorresponding(corr_dat, lat_dat, plt)
        opt.imgname = strcat(plt.name.aggr(1:end-2), 'trial_congruency');
        opt.size    = [45 20];
        opt.save    = plt.save;
        prepareFigure(fig.h, opt)
        close; clear fig opt plt_dat lat_dat corr_dat

    end

end
clear c curr_cond

% Exclude subjects based on defined criteria
idx_excld = logical(sum([isnan(exp.trialNo) ...
                         exp.prop.correspond_trials < 0.50], 2));

exp.trialNo(idx_excld, :)                = NaN;
exp.prop.val_trials(idx_excld, :)        = NaN;
exp.excl_trials(idx_excld, :)            = {[]};
exp.events.stim_onOff(idx_excld, :)      = {[]};
sacc.gazeShifts(idx_excld, :)            = {[]};
sacc.gazeShifts_zen(idx_excld, :)        = {[]};
sacc.time.resp(idx_excld, :)             = {[]};
sacc.time.fix(idx_excld, :)              = {[]};
sacc.time.search(idx_excld, :)           = {[]};
sacc.time.resp_bg(idx_excld, :)          = {[]};
sacc.propGs.closest(idx_excld, :)        = {[]};
sacc.propGs.further(idx_excld, :)        = {[]};
sacc.time.inspecting(idx_excld, :)       = {[]};
stim.chosenTarget(idx_excld, :)          = {[]};
stim.choiceCorrespond(idx_excld, :)      = {[]};
stim.no_easyDis(idx_excld, :)            = {[]};
stim.no_hardDis(idx_excld, :)            = {[]};
perf.score.final(idx_excld, :)           = NaN;
perf.hitMiss(idx_excld, :)               = {[]};
clear idx_excld


%% Print parameter regarding data quality and subjects performance
% Completed trials
clc; [round(mean(exp.trialNo, 'omitnan')); ...
      round(std(exp.trialNo, 'omitnan')); ...
      min(exp.trialNo); ...
      max(exp.trialNo)]

% Proportion valid trials
clc; round(mean(exp.prop.val_trials, 'omitnan').*100, 2)

% Proportion trials where response time could be calculated
clc; round(mean(exp.prop.resp_trials, 'omitnan').*100, 2)

% Proportion trials where last fixated and responded target corresponded
clc; round(mean(exp.prop.correspond_trials, 'omitnan').*100, 2)

% Final scores at end of conditions
perf.score.final(perf.score.final < 0) = 0;

clc; [round(mean(perf.score.final, 'omitnan'), 2); ...
      round(std(perf.score.final, 'omitnan'), 2); ...
      min(perf.score.final); ...
      max(perf.score.final)]
clc; [round(mean(sum(perf.score.final, 2), 'omitnan'), 2); ...
      round(std(sum(perf.score.final, 2), 'omitnan'), 2); ...
      min(sum(perf.score.final, 2)); ...
      max(sum(perf.score.final, 2))]
  

%% Distributions Euclidean distance between current fixation and closest stimulus
if plt.save == 1

    inp_dat = cellfun(@(x) x(:, [13 15 end]), sacc.gazeShifts(3:end, :), 'UniformOutput', false);
    infSampling_plt_distanceDist(inp_dat, plt)
    clear inp_dat

end

%% Proportion correct
fig.tit  = {'Single-target'; 'Double-target'};
fig.xLab = {'Easy target [proportion correct]'; [];};
fig.yLab = {'Hard target [proportion correct]'; [];};

perf.hitrates = NaN(exp.num.subNo, exp.num.condNo, 3);
fig.h         = figure;
for c = 1:exp.num.condNo % Condition

    % Proportion correct for individual subjects
    for s = 1:exp.num.subNo % Subject

        % Get proportion correct
        curr_sub         = exp.num.subs(s);
        inp_chosenTarget = stim.chosenTarget{curr_sub, c};
        inp_hitMiss      = perf.hitMiss{curr_sub, c};

        perf.hitrates(curr_sub, c, 1:3) = ...
            infSampling_propCorrect(inp_hitMiss, inp_chosenTarget);
        clear curr_sub inp_chosenTarget inp_hitMiss

    end
    clear s

end
opt.imgname = strcat(plt.name.aggr(1:end-2), 'propCorrect');
opt.size    = [35 15];
opt.save    = plt.save;
prepareFigure(fig.h, opt)
close; clear c fig opt


%% Proportion choices easy target
stim.propChoice.easy = NaN(9, exp.num.subNo, exp.num.condNo);
for c = 1:exp.num.condNo % Condition

    curr_cond = exp.num.conds(c);
    for s = 1:exp.num.subNo % Subject

        % Get data of subject
        curr_sub       = exp.num.subs(s);
        dat_sub_choice = stim.chosenTarget{curr_sub, c};
        dat_sub_ed     = stim.no_easyDis{curr_sub, c};

        % Get shown set-sizes
        ind_ss = unique(dat_sub_ed(~isnan(dat_sub_ed)));
        if curr_cond == 2 % Single-target condition

            % For the single-target condition, trials in which the easy
            % target was shown without distractors and trials in which the
            % difficult target was shown without distractors are coded the
            % same (i.e., "0" in the vector that indicates the trialwise
            % number of easy distractors). To calculate the proportion
            % choices here reliably, we have to set trials in which the
            % difficult target was chosen NaN
            dat_sub_ed(dat_sub_choice == 2) = NaN;

        end

        % For each set-size, determine proportion choices easy target
        no_ss = numel(ind_ss);
        for ss = 1:no_ss

            no_trials_val  = sum(dat_sub_ed == ind_ss(ss));
            no_trials_easy = sum(dat_sub_choice == stim.identifier(1, 1) & ...
                                 dat_sub_ed == ind_ss(ss));

            stim.propChoice.easy(ss, curr_sub, c) = no_trials_easy / no_trials_val;
            clear no_trials_val no_trials_easy

        end
        clear dat_sub_choice dat_sub_ed no_ss ss

        % Plot single-subject data
        if curr_cond == 3 % Double-target condition

            fig.h = figure;
            infSampling_plt_propChoiceEasy(stim.propChoice.easy(:, curr_sub, c), plt)
            opt.imgname = strcat(plt.name.aggr(1:end-2), 's', num2str(curr_sub), '_propChoices');
            opt.save    = plt.save;
            prepareFigure(fig.h, opt)
            close; clear fig opt l_h

        end
        clear curr_sub ind_ss

    end
    clear s

end
clear c curr_cond


%% Proportion gaze shifts on easy set as a function of set-size
sacc.propGs.onEasy_noLock_indSs = NaN(9, exp.num.subNo, exp.num.condNo);
for c = 1:exp.num.condNo % Condition

    curr_cond = exp.num.conds(c);
    for s = 1:exp.num.subNo % Subject

        % Get data of subject and drop excluded trials
        curr_sub = exp.num.subs(s);
        dat_sub  = sacc.gazeShifts{curr_sub, c};
        if ~isempty(dat_sub)

            % Get rid of entries we do not care about
            if curr_cond == 2 % Single-target condition

                % For the single-target condition, trials in which the easy
                % target was shown without distractors and trials in which
                % the difficult target was shown without distractors are
                % coded the same (i.e., "0" in the vector that indicates
                % the trialwise number of easy distractors). To calculate
                % the proportion choices here reliably, we have to set
                % trials in which the difficult target was chosen NaN
                dat_sub(dat_sub(:, 23) == 2, :) = NaN;

            end
%             dat_sub = dat_sub(dat_sub(:, 18) ~= 666, :);  % Gaze shifts on background
            dat_sub = dat_sub(~isnan(dat_sub(:, 23)), :); % Excluded trials & trials without choice

            % Get proportion gaze shifts on easy set, as a function of
            % set-size. For this, we use the same function we use to compute
            % the timecourse of proportion gaze shifts on different stimuli;
            % to get rid of the timecourse, we just assume each gaze shift
            % is the first one, so the function does not seperate between
            % different gaze shifts in a trial
            li_gsOnEasySet = any(dat_sub(:, 18) == stim.identifier(:, 1)', 2); % Get gaze shifts to easy set
            no_gs          = size(dat_sub, 1);
            inp_mat        = [ones(no_gs, 1) ...                                 Timelock; for this analysis, we have none
                              NaN(no_gs, 1) ...                                  Legacy column
                              dat_sub(:, 18) ...                                 AOI identifier
                              li_gsOnEasySet ...                                 Stimulus category of interest
                              dat_sub(:, 22) ...                                 Number easy distractors
                              dat_sub(:, 7) ...                                  Timestamp gaze shift offset
                              dat_sub(:, 26) ...                                 Trial number
                              dat_sub(:, 23)];                                 % Target chosen in trial
            clear no_gs li_gsOnEasySet dat_sub

            inp_minDat   = exp.avg.minDp;   % Minimum number datapoints to calculate mean
            inp_coiLab   = stim.identifier; % Category of interest == chosen target
            inp_ssGroups = (0:8)';          % Analyse each set-size seperately
            inp_lock     = 2;               % Arbitrary choice, since we do not assume any timecourse for this analysis
            propGs_onEasy_noLock_indSs = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups, inp_minDat);

            % Unpack and store data
            sacc.propGs.onEasy_noLock_indSs(:, curr_sub, c) = ...
                cell2mat(cellfun(@(x) x(:, 4, 3), propGs_onEasy_noLock_indSs, ...
                         'UniformOutput', false));
            clear inp_mat inp_minDat inp_coiLab inp_ssGroups inp_lock propGs_onEasy_noLock_indSs

        end
        clear curr_sub

    end
    clear s

    % Plot proportion gaze shifts on easy set against proportion choice for
    % easy target
    if curr_cond == 3 % Double-target condition

        % Unpack data
        prop_choiceEasy = stim.propChoice.easy(:, :, c);
        prop_choiceEasy = prop_choiceEasy(:);
        prop_saccEasy   = sacc.propGs.onEasy_noLock_indSs(:, :, c);
        prop_saccEasy   = prop_saccEasy(:);

        % Plot
        dat = [prop_choiceEasy prop_saccEasy];
        clear prop_choiceEasy prop_saccEasy

        fig.h = figure;
        infSampling_plt_corrChoiceEasyPropGsEasy(dat, plt)
        opt.imgname = strcat(plt.name.aggr(1:end-2), 'corr_propGsEasyPropChoiceEasy');
        opt.save    = plt.save;
        prepareFigure(fig.h, opt)
        clear fig opt dat; close 

    end

end
clear c curr_cond


%% Timecourse proportion gaze shifts on stimulus in trial
sacc.propGs.onChosen_trialBegin  = cell(exp.num.subNo, exp.num.condNo);
sacc.propGs.onEasy_trialBegin    = cell(exp.num.subNo, exp.num.condNo);
sacc.propGs.onSmaller_trialBegin = cell(exp.num.subNo, exp.num.condNo);
sacc.propGs.onCloser_trialBegin  = cell(exp.num.subNo, exp.num.condNo);
sacc.propGs.onChosen_lastSaccade = cell(exp.num.subNo, exp.num.condNo);
sacc.propGs.onEasy_lastSaccade   = cell(exp.num.subNo, exp.num.condNo);
for c = 2% 1:exp.num.condNo % Condition; only double-target

    curr_cond = exp.num.conds(c);
    for s = 1:exp.num.subNo % Subject

        % Get data of subject and drop excluded trials
        curr_sub = exp.num.subs(s);
        dat_sub  = sacc.gazeShifts{curr_sub, c};
        if ~isempty(dat_sub)

            % Get rid of entries we do not care about
%             dat_sub = dat_sub(dat_sub(:, 18) ~= 666, :); % Saccades on background
            dat_sub = dat_sub(~isnan(dat_sub(:, 23)), :);  % Excluded trials & trials without choice

            % Get stimulus category of interest, for which we want to
            % analyse proportion gaze shift that landed on it
            li_gsOnChosenSet   = any(dat_sub(:, 18) == stim.identifier(:, dat_sub(:, 23))', 2);                                       % Set of chosen target
            li_gsOnEasySet     = any(dat_sub(:, 18) == stim.identifier(:, 1)', 2);                                                    % Easy set
            li_gsOnSmallerSet  = (any(dat_sub(:, 18) == stim.identifier(:, 1)', 2) & dat_sub(:, 22) >= 0 & dat_sub(:, 22) <= 3) | ... % Smaller set
                                 (any(dat_sub(:, 18) == stim.identifier(:, 2)', 2) & dat_sub(:, 22) >= 5 & dat_sub(:, 22) <= 8) ;
            li_gsOnClosestStim = dat_sub(:, 21) == 1;                                                                                 % Closest stimulus

            % Get timecourse of proportion saccades on chosen set,
            % timelocked to trial start
            no_gs   = size(dat_sub, 1);
            inp_mat = [dat_sub(:, 24) ...   Timelock
                       NaN(no_gs, 1) ...    Legacy column
                       dat_sub(:, 18) ...   AOI identifier
                       li_gsOnChosenSet ... Stimulus category of interest
                       dat_sub(:, 22) ...   Number easy distractors
                       dat_sub(:, 7) ...    Timestamp gaze shift offset
                       dat_sub(:, 26) ...   Trial number
                       dat_sub(:, 23)];   % Target chosen in trial
            clear no_gs

            inp_minDat   = exp.avg.minDp;   % Minimum number datapoints to calculate mean
            inp_coiLab   = stim.identifier; % Category of interest == chosen target
            inp_ssGroups = 0:8;             % Analyse over all set-sizes
            inp_lock     = 2;               % Locked to trial beginning
            sacc.propGs.onChosen_trialBegin{curr_sub, c} = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups, inp_minDat);

            % Get timecourse of proportion gaze shifts on chosen set,
            % timelocked to last gaze shift in trial
            inp_mat(:, 1) = dat_sub(:, 25); % We just change the timelock and can otherwise resuse the input matrix
            inp_lock      = 1;              % Locked to last saccade in trial
            sacc.propGs.onChosen_lastSaccade{curr_sub, c} = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups, inp_minDat);
            clear li_gsOnChosenSet

            % Get timecourse of proportion gaze shifts on easy set,
            % timelocked to trial start
            inp_mat(:, [1 4]) = [dat_sub(:, 24) li_gsOnEasySet];
            inp_lock          = 2;
            sacc.propGs.onEasy_trialBegin{curr_sub, c} = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups, inp_minDat);

            % Get timecourse of proportion gaze shifts on easy set,
            % timelocked to last gaze shift in trial
            inp_mat(:, 1) = dat_sub(:, 25);
            inp_lock      = 1;
            sacc.propGs.onEasy_lastSaccade{curr_sub, c} = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups, inp_minDat);
            clear li_gsOnEasySet

            % Get timecourse of proportion gaze shifts on smaller set,
            % timelocked to first gaze shift in trial
            % timelocked to trial start
            inp_mat(:, [1 4]) = [dat_sub(:, 24) li_gsOnSmallerSet];
            inp_lock          = 2;
            sacc.propGs.onSmaller_trialBegin{curr_sub, c} = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups, inp_minDat);
            clear li_gsOnSmallerSet

            % Get timecourse of proportion gaze shifts on closer stimulus,
            % timelocked to first gaze shift in trial
            inp_mat(:, 4) = li_gsOnClosestStim;
            sacc.propGs.onCloser_trialBegin{curr_sub, c} = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups, inp_minDat);
            clear li_gsOnClosestStim inp_mat inp_minDat inp_coiLab inp_ssGroups inp_lock 

        end
        clear curr_sub dat_sub no_gs 

    end
    clear s

end
clear c curr_cond


%% Proportion gaze shifts to closest stimulus
sacc.propGs.closestFurther.overall = NaN(exp.num.subNo, exp.num.condNo, 2);
sacc.propGs.closestFurther.first   = NaN(exp.num.subNo, exp.num.condNo, 2);
for c = 1:exp.num.condNo % Condition

    curr_cond = exp.num.conds(c);
    for s = 1:exp.num.subNo % Subject

        curr_sub = exp.num.subs(s);
        dat_sub  = sacc.gazeShifts{curr_sub, c};
        if ~isempty(dat_sub)

            % Get rid of entries we do not care about
%             dat_sub = dat_sub(dat_sub(:, 18) ~= 666, :); % Saccades on background
            dat_sub = dat_sub(~isnan(dat_sub(:, 23)), :);  % Excluded trials & trials without choice

            % Overall proportion saccades to closest/further away stimulus
            sacc.propGs.closestFurther.overall(curr_sub, c, 1) = nanmean(sacc.propGs.closest{curr_sub, c});
            sacc.propGs.closestFurther.overall(curr_sub, c, 2) = nanmean(sacc.propGs.further{curr_sub, c});

            % Proportion first saccade to closest/further away stimulus
            li_firstAll     = dat_sub(:, 24) == 1;
            li_firstClosest = dat_sub(:, 24) == 1 & dat_sub(:, 21) == 1 & dat_sub(:, 18) ~= 666;
            li_firstFurther = dat_sub(:, 24) == 1 & dat_sub(:, 21) == 0 & dat_sub(:, 18) ~= 666;

            sacc.propGs.closestFurther.first(curr_sub, c, 1) = sum(li_firstClosest) / sum(li_firstAll);
            sacc.propGs.closestFurther.first(curr_sub, c, 2) = sum(li_firstFurther) / sum(li_firstAll);
            clear li_firstAll li_firstClosest li_firstFurther

        end
        clear curr_sub dat_sub

    end
    clear s

    % Plot aggregated data
    if curr_cond == 3

        plt_dat    = {sacc.propGs.closestFurther.overall(:, 2, :); ...
                      sacc.propGs.closestFurther.first(:, 2, :)};
        plt_x_lab  = {'Proportion gaze shifts to closest stimulus'; []};
        plt_y_lab  = {'Proportion gaze shifts to distant stimulus'; []};
        plt_sp_tit = {'Overall'; 'First saccade'};

        fig.h = figure;
        no_sp = numel(plt_dat);
        for sp = 1:no_sp % Subplot

            subplot(1, 2, sp)
            infSampling_plt_scatterPropSaccOnStim(plt_dat{sp}, ...
                                                  plt_x_lab{sp}, plt_y_lab{sp}, plt_sp_tit{sp}, ...
                                                  plt)

        end
        clear no_sp sp plt_dat plt_x_lab plt_y_lab plt_sp_tit

        opt.imgname = strcat(plt.name.aggr(1:end-2), 'propSacc_prox');
        opt.size    = [35 15];
        opt.save    = plt.save;
        prepareFigure(fig.h, opt)
        close; clear fig opt

    end

end
clear c curr_cond


%% Relationship between latency of first gaze shift in trial and first targeted AOI in trial
sacc.lat.firstGs_chosenSet   = NaN(exp.num.subNo, 2, exp.num.condNo);
sacc.lat.firstGs_easySet     = NaN(exp.num.subNo, 2, exp.num.condNo);
sacc.lat.firstGs_smallerSet  = NaN(exp.num.subNo, 2, exp.num.condNo);
sacc.lat.firstGs_closestStim = NaN(exp.num.subNo, 2, exp.num.condNo);
for c = 2% 1:exp.num.condNo % Condition; only double-target

    curr_cond = exp.num.conds(c);
    for s = 1:exp.num.subNo % Subject

        % Get data of single subject
        curr_sub = exp.num.subs(s);
        dat_sub  = sacc.gazeShifts{curr_sub, c};
        if ~isempty(dat_sub)

            % Get rid of entries we do not care about
    %         dat_sub = dat_sub(dat_sub(:, 18) ~= 666, :); % Saccades on background
            dat_sub = dat_sub(~isnan(dat_sub(:, 23)), :);  % Excluded trials & trials without choice

            % Get stimulus category of interest, for which we want to
            % analyse latencies
            li_gsOnChosenSet   = any(dat_sub(:, 18) == stim.identifier(:, dat_sub(:, 23))', 2);                                       % Set of chosen target
            li_gsOnEasySet     = any(dat_sub(:, 18) == stim.identifier(:, 1)', 2);                                                    % Easy set
            li_gsOnSmallerSet  = (any(dat_sub(:, 18) == stim.identifier(:, 1)', 2) & dat_sub(:, 22) >= 0 & dat_sub(:, 22) <= 3) | ... % Smaller set
                                 (any(dat_sub(:, 18) == stim.identifier(:, 2)', 2) & dat_sub(:, 22) >= 5 & dat_sub(:, 22) <= 8) ;
            li_gsOnClosestStim = dat_sub(:, 21) == 1;                                                                                 % Closest stimulus

            % Latencies of first gaze shift to chosen/not-chosen set
            inp_latencis = dat_sub(:, 11);   % Latencies
            inp_gsNos    = dat_sub(:, 24);   % Position of gaze shift in sequence
            inp_gsOnSOI  = li_gsOnChosenSet; % Gaze shift landed on stimulus category of interest or not
            inp_trialNos = dat_sub(:, 26);   % Trial numbers
            inp_noOI     = 1;                % Which gaze shift in sequence to analyse

            [~, sacc.lat.firstGs_chosenSet(curr_sub, 1:2, c)] = ...
                infSampling_getLatFirstGs(inp_latencis, inp_gsNos, inp_gsOnSOI, inp_trialNos, inp_noOI);
            clear li_gsOnChosenSet

            % Latencies of first gaze shift to easy/hard set
            inp_gsOnSOI = li_gsOnEasySet;

            [~, sacc.lat.firstGs_easySet(curr_sub, 1:2, c)] = ...
                infSampling_getLatFirstGs(inp_latencis, inp_gsNos, inp_gsOnSOI, inp_trialNos, inp_noOI);
            clear li_gsOnEasySet

            % Latencies of first gaze shift to smaller/larger set
            inp_gsOnSOI = li_gsOnSmallerSet;

            [~, sacc.lat.firstGs_smallerSet(curr_sub, 1:2, c)] = ...
                infSampling_getLatFirstGs(inp_latencis, inp_gsNos, inp_gsOnSOI, inp_trialNos, inp_noOI);
            clear li_gsOnSmallerSet

            % Latencies of first gaze shift to closest/further away stimulus
            inp_gsOnSOI = li_gsOnClosestStim;

            [~, sacc.lat.firstGs_closestStim(curr_sub, 1:2, c)] = ...
                infSampling_getLatFirstGs(inp_latencis, inp_gsNos, inp_gsOnSOI, inp_trialNos, inp_noOI);
            clear li_gsOnClosestStim inp_latencis inp_gsNos inp_gsOnSOI inp_trialNos inp_noOI

        end
        clear curr_sub dat_sub

    end
    clear s

end
clear c curr_cond


%% Search and non-search time
sacc.time.mean.fix        = NaN(exp.num.subNo, exp.num.condNo, 3);
sacc.time.mean.search     = NaN(exp.num.subNo, exp.num.condNo, 3);
sacc.time.mean.resp       = NaN(exp.num.subNo, exp.num.condNo, 3);
sacc.time.mean.non_search = NaN(exp.num.subNo, exp.num.condNo, 3);
for c = 1:exp.num.condNo % Condition

    curr_cond = exp.num.conds(c);
    for s = 1:exp.num.subNo % Subject

        curr_sub = exp.num.subs(s);

        % Get trials in which easy/hard target was chosen
        dat_chosenTarg_sub = stim.chosenTarget{curr_sub, c};
        li_chosen_easy     = dat_chosenTarg_sub == stim.identifier(1, 1);
        li_chosen_hard     = dat_chosenTarg_sub == stim.identifier(1, 2);
        clear dat_chosenTarg_sub

        % Fixation time
        sacc.time.mean.fix(curr_sub, c, :) = ...
            cat(3, ...
                nanmean(sacc.time.fix{curr_sub, c}), ...
                nanmean(sacc.time.fix{curr_sub, c}(li_chosen_easy)), ...
                nanmean(sacc.time.fix{curr_sub, c}(li_chosen_hard)));

        % Search time
        sacc.time.mean.search(curr_sub, c, :) = ...
            cat(3, ...
                nanmean(sacc.time.search{curr_sub, c}), ...
                nanmean(sacc.time.search{curr_sub, c}(li_chosen_easy)), ...
                nanmean(sacc.time.search{curr_sub, c}(li_chosen_hard)));

        % Response time
        sacc.time.mean.resp(curr_sub, c, :) =...
            cat(3, ...
                nanmean(sacc.time.resp{curr_sub, c}), ...
                nanmean(sacc.time.resp{curr_sub, c}(li_chosen_easy)), ...
                nanmean(sacc.time.resp{curr_sub, c}(li_chosen_hard)));

        % Non-search time
        non_search      = sacc.time.fix{curr_sub, c} + sacc.time.resp{curr_sub, c};
        non_search_easy = sacc.time.fix{curr_sub, c}(li_chosen_easy) + ...
                          sacc.time.resp{curr_sub, c}(li_chosen_easy);
        non_search_hard = sacc.time.fix{curr_sub, c}(li_chosen_hard) + ...
                          sacc.time.resp{curr_sub, c}(li_chosen_hard);

        sacc.time.mean.non_search(curr_sub, c, :) = ...
            cat(3, ...
                nanmean(non_search), ...
                nanmean(non_search_easy), ...
                nanmean(non_search_hard));
        clear curr_sub non_search non_search_easy non_search_hard li_chosen_easy li_chosen_hard

    end
    clear s

end
clear c
  

%% Analyse how much time participants spent searching for targets
sacc.time.inspecting_ss = NaN(exp.num.subNo, 9);
sacc.propGs.onAOI       = NaN(exp.num.subNo, 5);
for c = 1 % Single-target condition

    for s = 1:exp.num.subNo % Subject

        curr_sub = exp.num.subs(s);
        gs_sub   = sacc.gazeShifts{curr_sub, c};
        if ~isempty(gs_sub)

            inp_fixStim  = sacc.gazeShifts{curr_sub, c}(:, 18);
            inp_inspTime = sacc.time.inspecting{curr_sub, c};

            % Calculate mean inspection time for set sizes
            no_ss = unique(inp_inspTime(:, 3));
            no_ss = no_ss(~isnan(no_ss));
            for ss = 1:max(no_ss) % Set size

                li_trials = inp_inspTime(:, 3) == ss;
                sacc.time.inspecting_ss(s, ss+1) = mean(inp_inspTime(li_trials, 2), 'omitnan');
                clear li_trials

            end
            clear ss no_ss inp_inspTime

            % Analys proportion gaze shifts eye/difficult targets/distractors as well as background
            no_fixStim = unique(inp_fixStim);
            no_fixStim = no_fixStim(~isnan(no_fixStim));
            no_gs      = sum(~isnan(inp_fixStim));
            for fs = 1:numel(no_fixStim) % Fixated stimulus

                li_gs = inp_fixStim == no_fixStim(fs);
                sacc.propGs.onAOI(s, fs) = sum(li_gs) / no_gs;
                clear li_gs

            end
            clear fs inp_fixStim no_fixStim no_gs

        end
        clear curr_sub gs_sub

    end
    clear s

end
clear c

% Fit regression to inspection time
% lm_inspecTime = fitlm(0:8, mean(inspTime_ss, 'omitnan'));
no_lvl = size(sacc.time.inspecting_ss, 2);
no_sub = exp.num.subNo;
no_dis = 0;

sacc.time.inspecting_ss_long = [];
for lvl = 1:no_lvl % # distractors

    sacc.time.inspecting_ss_long = [sacc.time.inspecting_ss_long; ...
                                    zeros(no_sub, 1)+no_dis sacc.time.inspecting_ss(:, lvl)];

    no_dis = no_dis + 1;

end
sacc.time.inspecting_ss_fit = fitlm(sacc.time.inspecting_ss_long(:, 1), sacc.time.inspecting_ss_long(:, 2));


%% Export data for model
% Model scripts are build around getting data from exported .txt files and
% fitting the model to the imported data. To make things easier, I will
% keep this workflow, instead of "properly" implementing the model scripts
% into my framework
container_dat_label = infSampling_colNames;
dat_filenames       = {'dataSingleTarget.txt'  'dataDoubleTarget.txt'; ...
                       'dataSingleTarget.xlsx' 'dataDoubleTarget.xlsx'};
for c = 1:exp.num.condNo % Condition

    % Gather data to export
    container_dat = [exp.num.subs ...                                         1:     Subject numbers
                     reshape(perf.hitrates(:, c, :), exp.num.subNo, 3, 1) ...        Proportion correct (overall, easy, difficult)
                     NaN(exp.num.subNo, 60) ...                               5:64:  Placeholder for legacy columns (used to mean # saccades on stimulus, with and without set-size separation)
                     sacc.time.mean.search(:, c, 1) ...                       65:    Overall mean search time
                     sacc.time.mean.non_search(:, c, 1) ...                          Overall mean non-search time
                     sacc.time.mean.fix(:, c, 1), ...                                Overall mean fixation time
                     sacc.time.mean.resp(:, c, 1) ...                                Overall mean response time
                     sacc.time.mean.search(:, c, 2) ...                              Mean search time easy target chosen
                     sacc.time.mean.non_search(:, c, 2) ...                   70:    Mean non-search time easy target chosen
                     sacc.time.mean.resp(:, c, 2) ...                                Mean response time easy target chosen
                     sacc.time.mean.fix(:, c, 2), ...                                Mean fixation time easy target chosen
                     sacc.time.mean.search(:, c, 3) ...                              Mean search time difficult target chosen
                     sacc.time.mean.non_search(:, c, 3) ...                          Mean non-search time difficult target chosen
                     sacc.time.mean.resp(:, c, 3) ...                         75:    Mean response time difficult target chosen
                     sacc.time.mean.fix(:, c, 3) ...                                 Mean fixation time difficult target chosen
                     NaN(exp.num.subNo, 2) ...                                77:78: Placeholder for legacy columns (overall mean proportion choices for easy/hard target)
                     stim.propChoice.easy(:, :, c)' ...                       79:87: Proportion choices easy target as a function of set-size
                     1-stim.propChoice.easy(:, :, c)' ...                     88:96: Proportion choices difficult target as a function of set-size (will be all zeros in single-target condition, since this one was not calculated separately)
                     exp.trialNo(:, c)];                                      %      # solves trials
    container_dat = round(container_dat, 2);

    % Export data
    if exp.flag.export == 1

        % Save as .txt file
        dlmwrite(strcat(exp.name.analysis, '/_model/', dat_filenames{1, c}), ...
                 container_dat, ...
                 'delimiter', '\t')

        % Save as .xls file
        container_dat_xls = num2cell(container_dat);
        container_dat_xls(isnan(container_dat)) = {'NaN'};
        dat_table = array2table(container_dat_xls, ...
                                'VariableNames', container_dat_label');
        writetable(dat_table, ...
                   strcat(exp.name.analysis, '/_model/', dat_filenames{2, c}))

    end
    clear container_dat container_dat_xls

end
clear c container_dat_label dat_filenames


%% Fit model
cd(strcat(exp.name.analysis, '/_model'))
all = [];

% Get data from .xls files
all = get_params(all);
all = read_data(all);
plot_data(all);

% Fit model and plot results
all = fit_model(all);
plot_paper(all);
plot_proposal(all);
plot_model(all);
% plot_presentation(all); % IRTG retreat presentation

% Fit regression and plot results
all = fit_regression(all);
plot_regression(all);
% save('model.mat','all');


%% Create plots for paper
% Figure 2
% Proportion gaze shifts on different AOIs, search time as a function of
% distractor number, perceptual performance and temporal aspects of search
% behavior (planning-, search- and decision-time)
inp_dat_var      = cat(3, ...
                       [perf.hitrates(:, 1, 2)         perf.hitrates(:, 1, 3)], ...
                       [sacc.time.mean.fix(:, 1, 2)    sacc.time.mean.fix(:, 1, 3)], ...
                       [sacc.time.mean.search(:, 1, 2) sacc.time.mean.search(:, 1, 3)], ...
                       [sacc.time.mean.resp(:, 1, 2)   sacc.time.mean.resp(:, 1, 3)]);
inp_dat_reg      = [(0:8)' sacc.time.inspecting_ss'];
inp_dat_reg_long = sacc.time.inspecting_ss_long;
inp_mod_reg      = sacc.time.inspecting_ss_fit;
inp_dat_gs       = [(1:5)' sacc.propGs.onAOI'];

infSampling_plt_figTwo(inp_dat_reg, inp_dat_reg_long, inp_mod_reg, inp_dat_gs, inp_dat_var, plt)
clc; [round(mean(sacc.propGs.onAOI, 'omitnan'), 2); ...
      round(mean(sacc.propGs.onAOI, 'omitnan')-ci_mean(sacc.propGs.onAOI), 2); ...
      round(mean(sacc.propGs.onAOI, 'omitnan')+ci_mean(sacc.propGs.onAOI), 2)]                 % Proportion gaze shifts on AOI
clc; [round(mean(sacc.time.inspecting_ss, 'omitnan'), 2); ...
      round(mean(sacc.time.inspecting_ss, 'omitnan')-ci_mean(sacc.time.inspecting_ss), 2); ... % Search time as a function of set size
      round(mean(sacc.time.inspecting_ss, 'omitnan')+ci_mean(sacc.time.inspecting_ss), 2)]
clc; anova(sacc.time.inspecting_ss_fit, 'summary')
clc; matlab_pairedTtest(perf.hitrates(:, 1, 2), perf.hitrates(:, 1, 3))                        % Discrimination difficulty
clc; matlab_pairedTtest(sacc.time.mean.fix(:, 1, 2),    sacc.time.mean.fix(:, 1, 3));          % Planning time
clc; matlab_pairedTtest(sacc.time.mean.search(:, 1, 2), sacc.time.mean.search(:, 1, 3));       % Search time
clc; matlab_pairedTtest(sacc.time.mean.resp(:, 1, 2),   sacc.time.mean.resp(:, 1, 3));         % Decision time
clear inp_dat_var inp_dat_reg inp_dat_reg_long inp_mod_reg inp_dat_gs

% Figure 3
% Representative participants choices and regression as well as intercepts
% and slopes
prop_choices_easy     = stim.propChoice.easy(:, :, 2);
prop_choices_easy_fit = cat(3, all.reg.xn-1, all.reg.yn);
slopesIntercepts      = all.reg.fit;

infSampling_plt_figThree(prop_choices_easy, prop_choices_easy_fit, slopesIntercepts, plt)
clc; matlab_oneSampleTtest(all.reg.fit(:, 1)); % Intercepts
clc; matlab_oneSampleTtest(all.reg.fit(:, 2)); % Slopes
clear prop_choices_easy prop_choices_easy_fit slopesIntercepts

% Figure 4
% Empirical/predicted gain and empirical/predicted choices easy target
gain_emp              = all.data.double.perf;
choice_emp            = all.data.double.choices;
gain_mod_comb_perfect = all.model.perf_perfect(:, 3);
gain_mod_comb_noise   = all.model.perf(:, 3);
choice_mod_comb       = all.model.choices(:, :, 3);

infSampling_plt_figFour(gain_emp, gain_mod_comb_perfect, gain_mod_comb_noise, choice_emp, choice_mod_comb, plt)
clear gain_emp choice_emp gain_mod_comb_perfect gain_mod_comb_noise choice_mod_comb

% Figure 5
% Proportion gaze shifts on different stimuli over the course of trials
close all
inp_minSub = exp.avg.minSub;
inp_dat    = [sacc.propGs.onChosen_trialBegin(:, 2) ...
              sacc.propGs.onEasy_trialBegin(:, 2) ...
              sacc.propGs.onSmaller_trialBegin(:, 2) ...
              sacc.propGs.onCloser_trialBegin(:, 2)];

[inp_mean, inp_single] = infSampling_avgPropSacc(inp_dat, inp_minSub);
infSampling_plt_figFive(inp_mean, inp_single, plt)
clc; [round(mean(inp_single, 'omitnan'), 2); ...
      round(mean(inp_single, 'omitnan')-ci_mean(inp_single), 2); ...
      round(mean(inp_single, 'omitnan')+ci_mean(inp_single), 2)]                 % Proportion gaze shifts on AOI
clear inp_mean inp_single
% t = table((1:19)', inp_single(:, 1, 1), inp_single(:, 2, 1), inp_single(:, 3, 1), inp_single(:, 4, 1), ...
%           inp_single(:, 5, 1), inp_single(:, 6, 1), inp_single(:, 7, 1), inp_single(:, 8, 1), ...
%           'VariableNames', {'subNo', 'gs1', 'gs2', 'gs3', 'gs4', 'gs5', 'gs6', 'gs7', 'gs8'});
% Meas = table([1 2 3 4 5 6 7 8]', 'VariableNames', {'Measurements'});
% rm = fitrm(t, 'gs1-gs8 ~ 1', 'WithinDesign', Meas);
% % ranovatbl = ranova(rm)
% for d = 1:4 % Data
% 
%     data_long = [];
%     for s = 1:19 % Subject
% 
%         dat_sub   = inp_single(s, :, d)';
%         no_sub    = zeros(numel(dat_sub), 1)+s;
%         no_gs     = (1:numel(dat_sub))';
% %         data_long = [data_long; no_sub no_gs round(dat_sub.*100)];
%         data_long = [data_long; no_sub no_gs dat_sub];
% 
%     end
% %     data_long(isnan(data_long(:, 3)), 3) = 99;
% %     writematrix(data_long, strcat('dat_', num2str(d), '.csv'), 'Delimiter', ',');
%     csvwrite(strcat('dat_', num2str(d), '.csv'), data_long)
% 
% end

% Figure 6
% Latencies of first gaze shifts to different stimuli
inp_dat = cat(3, ...
              sacc.lat.firstGs_chosenSet(:, :, 2), ...
              sacc.lat.firstGs_easySet(:, :, 2), ...
              sacc.lat.firstGs_smallerSet(:, :, 2), ...
              sacc.lat.firstGs_closestStim(:, :, 2));
infSampling_plt_figSix(inp_dat, plt)
clc; matlab_pairedTtest(sacc.lat.firstGs_chosenSet(:, 1, 2), sacc.lat.firstGs_chosenSet(:, 2, 2))     % To chosen/not-chosen set
clc; matlab_pairedTtest(sacc.lat.firstGs_easySet(:, 1, 2), sacc.lat.firstGs_easySet(:, 2, 2))         % To easy/difficult set
clc; matlab_pairedTtest(sacc.lat.firstGs_smallerSet(:, 1, 2), sacc.lat.firstGs_smallerSet(:, 2, 2))   % To smaller/larger set
clc; matlab_pairedTtest(sacc.lat.firstGs_closestStim(:, 1, 2), sacc.lat.firstGs_closestStim(:, 2, 2)) % To closest/more distant stimulus
clear inp_dat