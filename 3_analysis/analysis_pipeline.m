close all; clear all; clc;


%% Experiment structure
exper.num.conds  = [2; 3]; % 2 == single-target, 3 == double-target
exper.num.subs   = (1:8)';
exper.num.subNo  = numel(exper.num.subs);
exper.num.condNo = numel(exper.num.conds);
if numel(exper.num.conds) < 2 || diff(exper.num.conds) ~= 1

    error('Run analysis with data from both single- and double-target condition')

end


%% Go to folder with data
exper.name.root = '/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual';
exper.name.analysis = strcat(exper.name.root, '/', '3_analysis');
exper.name.data = strcat(exper.name.root, '/', '2_data');
if exper.num.conds(1) == 2
    exper.name.plt = strcat(exper.name.root, '/', '4_figures/eye');
elseif exper.num.conds(1) == 4
    exper.name.plt = strcat(exper.name.root, '/', '4_figures/tablet');
end

addpath(exper.name.analysis, strcat(exper.name.analysis, '/_model'));
cd(exper.name.data);


%% Plot settings
plt.name.aggr     = strcat(exper.name.plt, '/infSampling_c_'); % Naming scheme; aggregated data
plt.color         = plotColors;                              % Default colors
plt.color.o1      = [241 163 64]./255;                       % Some additional colors
plt.color.o2      = [239 209 127]./255;
plt.color.p1      = [153 142 195]./255;
plt.color.p2      = [199 187 245]./255;
plt.color.mid     = [219 198 208]./255;
plt.color.dark    = [219-50 198-50 208-50]./255;
plt.size.mrk_ss   = 10;                                       % Markersize when plotting single-subject data
plt.size.mrk_mean = 12;                                      % Markersize when plotting aggregated data
plt.lw.thick      = 3;                                       % Linewidth
plt.lw.thin       = 2;
plt.save          = 0;                                       % Toggle if plots should be saved to drive
show_figs         = 1;                                       % Toggle if figure should be shown during plotting
if show_figs == 1

    set(groot, 'DefaultFigureVisible', 'on')

else

    set(groot, 'DefaultFigureVisible', 'off')

end
clear show_figs


%% Miscellaneous settings
exper.avg.minSub = 1; % Minimum number of subjects required to calculate mean
exper.flag.export = 1; % Export data for model
exper.crit.minDur = 5; % Minimum gaze shift duration; used for gaze shift detection
exper.name.export = [{'dataSingleTargetEye',    'dataDoubleTargetEye'}; ...
                     {'',                       ''}; ... Add an empty column for indexing to work
                     {'dataSingleTargetTablet', 'dataDoubleTargetTablet'}];


%% Settings of screen, on which data was recorded
screen = screenBig;
screen.fixTol = 1.5;

% Define position of fixation cross
exper.fixLoc.deg = [0, 9.5];
exper.fixLoc.px = round([(exper.fixLoc.deg(1) / screen.xPIX2DEG) + screen.x_center ... 
                         (exper.fixLoc.deg(2) / screen.yPIX2DEG) + screen.y_center]);


%% Stimulus settings
% We define our AOI as a circular area, with a diameter of 5deg, around the
% center of each stimulus
stim.diameter.px = 49; % Stimulus diameter (pixel)
stim.diameter.deg = stim.diameter.px*screen.xPIX2DEG; % Stimulus diameter (deg)
stim.diameterAOI.deg = 3;
stim.radiusAOI.deg = stim.diameterAOI.deg/2; % AOI radius (deg)

% Identifier for easy (:, 1) and hard (:, 2) targets (1, :) and distractors (2, :)
stim.identifier = [1, 2; 3, 4];
stim.identifier_bg = 666;


%% Define columns for log file
log.col.trialNo      = 4;     % Trial #
log.col.noTargets    = 5;     % # of targets in trial
log.col.targetDiff   = 6;     % Target type shown in trial (only single-target experiment); 1 == easy, 2 == difficult
log.col.diffLvlEasy  = 7;     % Difficulty level of easy target
log.col.diffLvlHard  = 8;     % Difficulty level of difficult target
log.col.gapPosEasy   = 9;     % Gap location on easy target; 1 == bottom, 2 == top, 3 == left, 4 == right
log.col.gapPosHard   = 10;    % Gap location on difficult target
log.col.gapPosReport = 11;    % Reported gap location
log.col.noDisEasy    = 12;    % # of easy distractors in trial
log.col.noDisHard    = 13;    % # of difficult distractors in trial
log.col.noDisOverall = 14;    % Overall # of distractors in trial
log.col.stimPosX     = 15:24; % Positions on x-axis
log.col.stimPosY     = 25:34; % Positions on y_axis
log.col.cumTimer     = 35;    % Cumulative timer
log.col.hitMiss      = 36;    % Hit/miss
log.col.score        = 37;    % Number of points
log.col.fixErr       = 38;    % Flag for fixation error


%% Get data from trials
exper.trialNo           = NaN(exper.num.subNo, exper.num.condNo);  % # of solved trials
exper.excl_trials       = cell(exper.num.subNo, exper.num.condNo); % Trials with fixation error
exper.events.stim_onOff = cell(exper.num.subNo, exper.num.condNo); % Timestamps of stimulus on- and offset
exper.cum_trialTime     = cell(exper.num.subNo, exper.num.condNo); % Cumulative time spent on a trial
sacc.gazeShifts         = cell(exper.num.subNo, exper.num.condNo); % Gaze shifts (blinks and saccades)
sacc.gazeShifts_zen     = cell(exper.num.subNo, exper.num.condNo); % Gaze shifts (blinks and saccades) for Zenodo
sacc.time.planning      = cell(exper.num.subNo, exper.num.condNo); % Planning times
sacc.time.inspection    = cell(exper.num.subNo, exper.num.condNo); % Inspection times
sacc.time.decision      = cell(exper.num.subNo, exper.num.condNo); % Decision times
sacc.time.resp_bg       = cell(exper.num.subNo, exper.num.condNo); % Time between last gaze shift on background and response
sacc.time.search        = cell(exper.num.subNo, exper.num.condNo); % Time that was spent searching targets in a trial
sacc.propGs.closest     = cell(exper.num.subNo, exper.num.condNo); % Proportion gaze shifts to closest stimulus
sacc.propGs.further     = cell(exper.num.subNo, exper.num.condNo); % Proportion gaze shifts to further away stimulus
sacc.propGs.aoiFix      = cell(exper.num.subNo, exper.num.condNo); % Flags if at least one defined AOI was fixated in a trial
stim.chosenTarget       = cell(exper.num.subNo, exper.num.condNo); % Chosen target
stim.choiceCorrespond   = cell(exper.num.subNo, exper.num.condNo); % Corresponde chosen and last saccade target
stim.no_easyDis         = cell(exper.num.subNo, exper.num.condNo); % # easy distractors
stim.no_hardDis         = cell(exper.num.subNo, exper.num.condNo); % # difficult distractors
perf.score.final        = NaN(exper.num.subNo, exper.num.condNo);  % Score at end of condition
perf.hitMiss            = cell(exper.num.subNo, exper.num.condNo); % Hit/miss in trial
for c = 1:exper.num.condNo % Condition

    curr_cond = exper.num.conds(c);
    for s = 1:exper.num.subNo % Subject

        % Go to folder of single subject
        curr_sub    = exper.num.subs(s);
        dirName_sub = dir(strcat(sprintf('e%dv%db%d', curr_cond, curr_sub), '*'));
        if ~isempty(dirName_sub)
            dirName_sub = dirName_sub(end).name;
            cd(dirName_sub);
        else
            disp(['Skipping missing participant ', num2str(curr_sub)])
            continue
        end

        % Load .log file of single subject and extract relevant data from it
        fileName_log = [dirName_sub, '.log'];
        log.file     = readmatrix(fileName_log);
        if log.file(1, 4) > 1 % Recode trial #
            keyboard
            log.file(:, 4) = log.file(:, 4) - min(log.file(:, 4)) + 1;
        end
        if mod(curr_cond, 2) == 0 % Single-target condition

            % For the single-target condition, trials in which the easy
            % target was shown without distractors and trials in which the
            % difficult target was shown without distractors are coded the
            % same (i.e., "0" in the vector that indicates the trialwise
            % number of easy distractors). To make things unambigous, we
            % have to set the # easy distractors NaN for trials where the
            % difficult target was chosen (and we have to do the same with
            % # difficult distractors and trials where the easy target was
            % chosen)
            li_choiceEasy      = log.file(:, log.col.targetDiff) == stim.identifier(1, 1);
            li_choiceDifficult = log.file(:, log.col.targetDiff) == stim.identifier(1, 2);

            log.file(li_choiceDifficult, log.col.noDisEasy) = NaN;
            log.file(li_choiceEasy, log.col.noDisHard)      = NaN;
            clear li_choiceEasy li_choiceDifficult

        end
        clear fileName_log dirName_sub

        exper.trialNo(curr_sub, c)     = max(log.file(:, log.col.trialNo));      % # completed trials
        exper.excl_trials{curr_sub, c} = find(log.file(:, log.col.fixErr) == 1); % Trials with fixation error
        perf.score.final(curr_sub, c)  = log.file(end, log.col.score);           % Score of subject at end of condition
        perf.hitMiss{curr_sub, c}      = log.file(:, log.col.hitMiss);           % Hit/miss in trial
        stim.no_easyDis{curr_sub, c}   = log.file(:, log.col.noDisEasy);         % # easy distractors in trial
        stim.no_hardDis{curr_sub, c}   = log.file(:, log.col.noDisHard);         % # difficult distractors in trial

        % Iterate through trials and get gaze shifts, fixated AOIs and
        % search as well as non-search times
        no_trials_singleSub = exper.trialNo(curr_sub, c); % Number of trials

        trial.events.stim_onOff  = NaN(no_trials_singleSub, 2);   % Timestamps of stimulus on- and offset
        time_planning            = NaN(no_trials_singleSub, 1);   % Plannning times
        time_inspection          = NaN(no_trials_singleSub, 1);   % Inspection times
        time_decision            = NaN(no_trials_singleSub, 1);   % Decision times
        time_respBg              = NaN(no_trials_singleSub, 2);   % Time between last gaze shift on background and response
        time_trial               = NaN(no_trials_singleSub, 1);   % Time between stimulus onset and offset/response
        inspectedElements_no     = NaN(no_trials_singleSub, 5);   % # unique stimuli a subjects inspected in a trial
        choice_target            = NaN(no_trials_singleSub, 1);   % Chosen target
        choice_congruence        = NaN(no_trials_singleSub, 1);   % Correspondence responded and last fixated target
        prop_gsClosest           = NaN(no_trials_singleSub, 1);   % Proportion gaze shifts to closest AOI
        prop_gsFurther           = NaN(no_trials_singleSub, 1);   % Proportion gaze shifts to more distant AOI
        li_atLeastOneGs          = zeros(no_trials_singleSub, 1); % Flag if at least one gaze shift to any defined AOI was made in a trial
        gazeShifts_allTrials_zen = [];                            % Gaze shift matrix for Zenodo
        gazeShifts_allTrials     = [];                            % Gaze shift matrix for analysis
        for t = 1:no_trials_singleSub % Trial

            % Get gaze trace in trial and preprocess data
            if curr_cond < 4

                flip_yAxis = 1;

                [gTrace, flag_dataLoss]        = loadDat(t, screen.x_pix, screen.y_pix);
                exper.excl_trials{curr_sub, c} = [exper.excl_trials{curr_sub, c}; flag_dataLoss];
                if ~isempty(flag_dataLoss)
    
                    log.file(t, log.col.fixErr) = 1;
    
                end
                [gTrace(:, 2), gTrace(:, 3)] = ...
                    convertPix2Deg(gTrace(:, 2), gTrace(:, 3), ...
                                   [exper.fixLoc.px(1) exper.fixLoc.px(2)], ...
                                   [screen.xPIX2DEG  screen.yPIX2DEG], ...
                                   flip_yAxis);

                trial.gazeTrace = gTrace;
                clear flip_yAxis gTrace flag_dataLoss

            end

            % Get events in trial
            % We expect five events to happen in a trial:
            % trial begin, start recording, fixation onset, onset of stimuli
            % and offset of stimuli [i.e., response])
            if curr_cond < 4

                trial.events.all = find(bitget(trial.gazeTrace(:, 4), 3));
                if numel(trial.events.all) ~= 5

                    % For one participant in the single-target condition, we
                    % miss a chunk of data right at trial start, which results
                    % in an event being missing. To stop the analysis from
                    % crashing, we add a placeholder so that the event vector
                    % has the appropriate length
    %                 keyboard
                    trial.events.all = [trial.events.all(1:2); NaN; trial.events.all(3:end)];

                end
                trial.events.stim_onOff(t, 1) = trial.gazeTrace(trial.events.all(4), 1);                       % Timestamp stimulus onset
                trial.events.stim_onOff(t, 2) = trial.gazeTrace(trial.events.all(5), 1);                       % Timestamp stimulus offset
                time_trial(t)                 = trial.events.stim_onOff(t, 2) - trial.events.stim_onOff(t, 1); % Time spent on trial

            elseif curr_cond > 3

                fileName_events  = sprintf('e%dv%db1_events.csv', curr_cond, curr_sub);
                temp             = readmatrix(fileName_events);
                trial.events.all = temp(t, :)';

                trial.events.stim_onOff(t, 1) = trial.events.all(4);                       % Timestamp stimulus onset
                trial.events.stim_onOff(t, 2) = trial.events.all(5);                       % Timestamp stimulus offset
                time_trial(t)                 = trial.events.stim_onOff(t, 2) - trial.events.stim_onOff(t, 1); % Time spent on trial
                clear fileName_events

            end

            % Offline check for fixation errors (just to be sure)
            if curr_cond < 4

                fixPos_stimOn  = trial.gazeTrace(trial.events.all(4)-20:trial.events.all(4)+80, 2:3);
                fixPos_deviate = sum(abs(fixPos_stimOn(:)) > screen.fixTol);
                if fixPos_deviate > 0 && ~ismember(t, exper.excl_trials{curr_sub, c})

                    keyboard
                    exper.excl_trials{curr_sub, c} = [exper.excl_trials{curr_sub, c}; t];

                end
                clear fixPos_stimOn fixPos_deviate

            end

            % The stimulus locations, extracted from the .log-file, are not
            % ordered: because of this, different cells in the location
            % vector might correspond to different stimulus types, depending
            % on the number of easy/difficult distractors in a trial. To make
            % our life easier, we want to order them so that each position
            % in the location matrix is directly linked to one type of
            % stimulus (easy/difficult target/distractor)
            % CAUTION: SINCE WE FLIPPED THE Y-COORDINATES OF THE GAZE TRACE
            % WE ALSO FLIP THE Y-COORDINATES OF THE STIMULUS LOCATIONS
            inp_x_loc = log.file(t, log.col.stimPosX); % Stimulus locations
            inp_y_loc = log.file(t, log.col.stimPosY);
            inp_no_ed = log.file(t, log.col.noDisEasy); % # easy distractors
            inp_no_dd = log.file(t, log.col.noDisHard); % # difficult distractors
            inp_no_targ = log.file(t, log.col.noTargets); % Number targets in condition
            if mod(curr_cond, 2) == 0

                inp_shownTarg = log.file(t, log.col.targetDiff); % Target shown in trial

            elseif mod(curr_cond, 2) % In the double-target condition, both target were always shown

                inp_shownTarg = NaN;
                
            end

            stim_locations = ...                  
                infSampling_getStimLoc(inp_x_loc, inp_y_loc, ...
                                       inp_no_ed, inp_no_dd, ...
                                       inp_no_targ, inp_shownTarg);
            clear inp_x_loc inp_y_loc inp_no_ed inp_no_dd inp_no_targ inp_shownTarg

            % Get gaze shifts in trial
            % Gaze shifts are all saccades and blinks, detected in a trial
            if curr_cond < 4

                inp_gazeTrace = trial.gazeTrace(trial.events.all(4):trial.events.all(5), :); % Gaze trace between stimulus on- and offset
                inp_ts_stimOn = trial.events.stim_onOff(t, 1); % Timestampe stimulus onset
                inp_minDur_sacc = exper.crit.minDur; % Minimum duration of gaze shifts [ms]; everything beneath is flagged
                inp_screen_x = (screen.x_pix - exper.fixLoc.px(1)) * screen.xPIX2DEG; % Most extrem gaze position possible, given the display size
                inp_screen_y= [exper.fixLoc.px(2) * screen.yPIX2DEG ...
                               (screen.y_pix - exper.fixLoc.px(2)) * screen.yPIX2DEG*-1];

                gazeShifts_singleTrial = ...
                    infSampling_getGazeShifts(inp_gazeTrace, inp_ts_stimOn, ...
                                              inp_minDur_sacc, inp_screen_x, inp_screen_y);
                clear inp_gazeTrace inp_ts_stimOn inp_minDur_sacc inp_screen_x inp_screen_y

            elseif curr_cond > 3

                fileName_sacc  = sprintf('e%dv%db1_saccades.csv', curr_cond, curr_sub);
                temp           = readmatrix(fileName_sacc);
                if temp(1, 17) > 1

                    temp(:, 17) = temp(:, 17) - min(temp(:, 17)) + 1;

                end
                gazeShifts_singleTrial = temp(temp(:, 17) == t, :);
                gazeShifts_singleTrial = gazeShifts_singleTrial(:, 1:end-1);
                clear fileName_sacc

            end

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
            if ~isempty(gazeShifts_singleTrial)
                inp_endpoints_x = gazeShifts_singleTrial(:, 13); % Mean gaze position after gaze shift
                inp_endpoints_y = gazeShifts_singleTrial(:, 15);
                inp_stimLoc_x   = stim_locations(:, :, 1); % Stimulus locations
                inp_stimLoc_y   = stim_locations(:, :, 2);
                inp_aoi_radius  = stim.radiusAOI.deg; % Desired AOI size
                inp_debug_plot  = 0; % Plot stimulus locations and gaze shift endpoints
    
                gazeShifts_singleTrial(:, end+1) = ...
                    infSampling_getFixatedAOI(inp_endpoints_x, inp_endpoints_y, ...
                                              inp_stimLoc_x, inp_stimLoc_y, ...
                                              inp_aoi_radius, stim.identifier_bg, ...
                                              inp_debug_plot);
                clear inp_endpoints_x inp_endpoints_y inp_stimLoc_x inp_stimLoc_y inp_aoi_radius inp_debug_plot
            end

            % Currently, each individual distractor has an unique identifier,
            % which corresponds to the distractors location in the position
            % matrix; to make things easier, we assign easy/difficult
            % distractors an unambigous identifier each. For targets and
            % the background the identifier stays the same as before
            li_de = gazeShifts_singleTrial(:, end) > 2 & gazeShifts_singleTrial(:, end) <= 10;
            li_dd = gazeShifts_singleTrial(:, end) > 10 & gazeShifts_singleTrial(:, end) <= 18;
            li_d  = sum([li_de li_dd ], 2);

            gazeShifts_singleTrial(:, end+1)   = zeros;
            gazeShifts_singleTrial(li_de, end) = stim.identifier(2, 1);                % Easy distractor
            gazeShifts_singleTrial(li_dd, end) = stim.identifier(2, 2);                % Difficult distractor
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
            if ~isempty(gazeShifts_singleTrial)
                inp_gazeShifts = gazeShifts_singleTrial;
                inp_stimOn     = trial.events.stim_onOff(t, 1);
                inp_stimOff    = trial.events.stim_onOff(t, 2);
                if mod(curr_cond, 2) == 0
                    expNo = 2;
                elseif mod(curr_cond, 2) == 1
                    expNo = 3;
                end
                [time_planning(t), time_decision(t), time_inspection(t), gazeShifts_noConsec_singleTrial, time_respBg(t, :)] = ...
                    infSampling_getDwellTime(inp_gazeShifts, inp_stimOn, inp_stimOff, stim.identifier, stim.identifier_bg, expNo);
                clear inp_gazeShifts inp_stimOn inp_stimOff expNo
            else
                gazeShifts_noConsec_singleTrial = gazeShifts_singleTrial;
            end

            % Check if at least one gaze shift was made to any AOI
            if ~isempty(gazeShifts_noConsec_singleTrial) && ~all(gazeShifts_noConsec_singleTrial(:, 18) == stim.identifier_bg)

                li_atLeastOneGs(t) = 1;

            end

            % Check if gaze shifts went to closest stimulus
            euc_dist = NaN;
            if ~isempty(gazeShifts_noConsec_singleTrial)
                inp_gsOn_x    = gazeShifts_noConsec_singleTrial(:, 5);  % Onset position of gaze shifts
                inp_gsOn_y    = gazeShifts_noConsec_singleTrial(:, 6); 
                inp_targAoi   = gazeShifts_noConsec_singleTrial(:, 17); % Index of gaze shift target
                inp_flagBg    = stim.identifier_bg;                     % Flag, marking background as gaze shift target
                inp_stimLoc_x = stim_locations(:, :, 1);                % Stimulus locations
                inp_stimLoc_y = stim_locations(:, :, 2);
    
                [gazeShifts_noConsec_singleTrial(:, end+1), prop_gsClosest(t), prop_gsFurther(t)] = ...
                    infSampling_distStim(inp_gsOn_x, inp_gsOn_y, inp_targAoi, ...
                                         inp_stimLoc_x, inp_stimLoc_y, inp_flagBg);
    
                % Check distance between gaze while fixating and closest stimulus
                inp_gsOn_x  = gazeShifts_noConsec_singleTrial(:, 13); % Onset position of gaze shifts
                inp_gsOn_y  = gazeShifts_noConsec_singleTrial(:, 15);
                inp_targAoi = NaN(size(inp_gsOn_x, 1), 1);            % Do not correct for currently fixated AOI
    
                [~, ~, ~, euc_dist] = ...
                    infSampling_distStim(inp_gsOn_x, inp_gsOn_y, inp_targAoi, ...
                                         inp_stimLoc_x, inp_stimLoc_y, inp_flagBg);
                clear inp_gsOn_x inp_gsOn_y inp_targAoi inp_flagBg inp_stimLoc_x inp_stimLoc_y
            end

            % Get chosen target in trial
            % 1 == easy target, 2 == hard target
            if mod(curr_cond, 2) == 1 % Double-target condition

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
                inp_flag_bg     = stim.identifier_bg;

                [choice_target(t), ~, choice_congruence(t)] = ...
                    infSampling_getChosenTarget(inp_gapLoc_easy, inp_resp, inp_fixAOI, ...
                                                inp_flag_targ, inp_flag_dis, inp_flag_bg);
                clear inp_gapLoc_easy inp_resp inp_fixAOI inp_flag_targ inp_flag_dis inp_flag_bg

            elseif mod(curr_cond, 2) == 0 % Single-target condition

                % Chosen target is the target shown in trial
                choice_target(t) = log.file(t, log.col.targetDiff);

            end

            % Count how many unique stimuli were fixated in a trial,
            % determine how much time was spent searching in a trial (#
            % fixated stimuli * inspection time in a trial) and add #
            % distractors (of the chosen set) in a trial
            % If one and the same stimulus was fixated more than once, we
            % treat this as one fixation
            if ~isempty(gazeShifts_noConsec_singleTrial)
                inspectedElements_no(t, 1:3) = infSampling_getUniqueFixations(gazeShifts_noConsec_singleTrial(:, 17), ...
                                                                              stim.identifier(1, :), ...
                                                                              stim.identifier_bg, ...
                                                                              curr_cond);
                inspectedElements_no(t, 4) = time_trial(t);
                if choice_target(t) == stim.identifier(1, 1) % Easy chosen
                    inspectedElements_no(t, 5) = log.file(t, log.col.noDisEasy);
                elseif choice_target(t) == stim.identifier(1, 2) % Difficult chosen
                    inspectedElements_no(t, 5) = log.file(t, log.col.noDisHard);
                end
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
            if ~isempty(gazeShifts_noConsec_singleTrial)
                no_gs_ncs            = size(gazeShifts_noConsec_singleTrial, 1);
                li_noBg              = gazeShifts_noConsec_singleTrial(:, 18) ~= stim.identifier_bg;
                timelock             = NaN(no_gs_ncs, 2);
                timelock(li_noBg, 1) = (1:sum(li_noBg))';
                timelock(li_noBg, 2) = (sum(li_noBg)-1:-1:0)';
    
                gazeShifts_allTrials = [gazeShifts_allTrials; ...
                                        gazeShifts_noConsec_singleTrial ...                    Gaze shifts in trial 
                                        zeros(no_gs_ncs, 1)+log.file(t, log.col.noDisEasy) ... Number easy distractors
                                        zeros(no_gs_ncs, 1)+choice_target(t) ...               Chosen target
                                        timelock ...                                           Timelock to trial start/last gaze shift
                                        zeros(no_gs_ncs, 1)+t, ...                             Trial number
                                        euc_dist(:, end), ...                                  Distance to closest stimulus (includes the currently fixated stimulus)
                                        zeros(no_gs_ncs, 1)+log.file(t, log.col.noDisHard) ... Number difficult distractors
                                        zeros(no_gs_ncs, 1)+inspectedElements_no(t, 5)];     % Number distractors from chosen set
            end
            clear no_gs_ncs gazeShifts_noConsec_singleTrial timelock li_noBg euc_dist

            % Gaze shift matrix, used for export to Zenodo
            % This one contains all gaze shifts, including consecutive gaze
            % shifts and some additional data (set-size easy, chosen target, 
            % timestamps for stimulus on- and offset and trialnumber). For
            % the gaze shifts, we export timestamps and coordinates of on-
            % and offset, type of gaze shift (saccade/blink) and mean
            % and std of gaze between AOI visits
            if ~isempty(gazeShifts_singleTrial)
                no_gs = size(gazeShifts_singleTrial, 1);
                gapLocChosen = NaN(no_gs, 2);
                if mod(curr_cond, 2) == 0
                    gapLocChosen(:, choice_target(t)) = log.file(t, log.col.gapPosEasy);
                elseif mod(curr_cond, 2) == 1
                    gapLocChosen = repmat([log.file(t, log.col.gapPosEasy), log.file(t, log.col.gapPosHard)], ...
                                           no_gs, 1);
                end
                gazeShifts_allTrials_zen = [gazeShifts_allTrials_zen; ...
                                            zeros(no_gs, 1)+curr_cond-1, ...                        Condition number
                                            zeros(no_gs, 1)+curr_sub, ...                           Subject number
                                            gazeShifts_singleTrial(:, 1:2), ...                     Gaze shifts in trial
                                            zeros(no_gs, 1)+log.file(t, log.col.fixErr), ...        Exclude trial?
                                            gazeShifts_singleTrial(:, 3:16), ...
                                            zeros(no_gs, 1)+log.file(t, log.col.targetDiff), ...    Shown target (only single-target)
                                            zeros(no_gs, 1)+log.file(t, log.col.noDisEasy) ...      Number easy distractors
                                            zeros(no_gs, 1)+log.file(t, log.col.noDisHard) ...      Number difficult distractors
                                            zeros(no_gs, 1)+gapLocChosen(:, 1) ...                  Gap location on easy
                                            zeros(no_gs, 1)+gapLocChosen(:, 2) ...                  Gap location on difficult
                                            zeros(no_gs, 1)+log.file(t, log.col.gapPosReport) ...   Gap location reported
                                            zeros(no_gs, 1)+log.file(t, log.col.hitMiss) ...        Hit/miss
                                            zeros(no_gs, 1)+log.file(t, log.col.score) ...          Score after trial
                                            zeros(no_gs, 2)+trial.events.stim_onOff(t, :) ...       Timestamps stimulus on-/offset
                                            zeros(no_gs, 1)+t, ...                                  Trial number
                                            zeros(no_gs, 18)+stim_locations(:, :, 1), ...           x/y coordinates of stimulus locations
                                            zeros(no_gs, 18)+stim_locations(:, :, 2)];
            end
            clear no_gs gazeShifts_singleTrial stim_locations gapLocChosen

        end
        clear t no_trials_singleSub

        % Store data of subject
        exper.events.stim_onOff{curr_sub, c} = trial.events.stim_onOff;  % Timestamps of stimulus on- and offset
        exper.cum_trialTime{curr_sub, c}     = time_trial;               % Time spent on trial
        stim.chosenTarget{curr_sub, c}       = choice_target;            % Target, chosen in trial
        stim.choiceCorrespond{curr_sub, c}   = choice_congruence;        % Correspondece between last fixated and responded on target
        sacc.gazeShifts{curr_sub, c}         = gazeShifts_allTrials;     % Non-consecutive gaze shifts
        sacc.gazeShifts_zen{curr_sub, c}     = gazeShifts_allTrials_zen; % All gaze shifts (for Zenodo)
        sacc.time.planning{curr_sub, c}      = time_planning;            % Planning times (time between first saccade offset and stimulus onset)
        sacc.time.inspection{curr_sub, c}    = time_inspection;          % Inspection times (for each fixated stimulus, time between entering gaze
                                                                         % shift onset and time leaving gaze shift offset)
        sacc.time.decision{curr_sub, c}      = time_decision;            % Decision times (time between last saccade offset and response)
        sacc.time.resp_bg{curr_sub, c}       = time_respBg;              % Time between last gaze shift in background and response
        sacc.time.search{curr_sub, c}        = inspectedElements_no;     % # inspected elements & time spent searching for targets
        sacc.propGs.closest{curr_sub, c}     = prop_gsClosest;           % Proportion gaze shifts to closest AOI
        sacc.propGs.further{curr_sub, c}     = prop_gsFurther;           % Proportion gaze shifts to closest AOI
        sacc.propGs.aoiFix{curr_sub, c}      = li_atLeastOneGs;          % Flag if at least one defined AOI was fixated in a trial
        clear curr_sub trial gazeShifts_allTrials choice_target time_planning time_inspection time_decision gazeShifts_allTrials_zen
        clear prop_gsClosest prop_gsFurther choice_congruence time_respBg inspectedElements_no time_trial li_atLeastOneGs

        cd(exper.name.data);

    end
    clear s curr_cond

end
clear c log
cd(exper.name.root);


%% Export for Zenodo
% for c = 1:exper.num.condNo % Condition
% 
%     temp = vertcat(sacc.gazeShifts_zen{:, c});
%     writematrix(temp, strcat('./dat_cond', num2str(c), '.csv'))
% 
% end
% sacc = rmfield(sacc, 'gazeShifts_zen'); 


%% Exclude invalid trials and check data quality
exper.prop.val_trials     = NaN(exper.num.subNo, exper.num.condNo);
exper.timeLostExcldTrials = zeros(exper.num.subNo, exper.num.condNo);
exper.noExcludedTrial     = NaN(exper.num.subNo, exper.num.condNo);
sacc.propGs.aoiFix_mean   = NaN(exper.num.subNo, exper.num.condNo);
for c = 1:exper.num.condNo % Condition

    for s = 1:exper.num.subNo % Subject

        curr_sub  = exper.num.subs(s);
        idx_excld = sort(unique(exper.excl_trials{curr_sub, c}));

        exper.events.stim_onOff{curr_sub, c}(idx_excld, :) = NaN;
        sacc.time.planning{curr_sub, c}(idx_excld, :)      = NaN;
        sacc.time.inspection{curr_sub, c}(idx_excld, :)    = NaN;
        sacc.time.decision{curr_sub, c}(idx_excld, :)      = NaN;
        sacc.time.resp_bg{curr_sub, c}(idx_excld, :)       = NaN;
        sacc.time.search{curr_sub, c}(idx_excld, :)        = NaN;
        sacc.propGs.closest{curr_sub, c}(idx_excld, :)     = NaN;
        sacc.propGs.further{curr_sub, c}(idx_excld, :)     = NaN;
        sacc.propGs.aoiFix{curr_sub, c}(idx_excld, :)      = NaN;
        stim.chosenTarget{curr_sub, c}(idx_excld, :)       = NaN;
        stim.choiceCorrespond{curr_sub, c}(idx_excld, :)   = NaN;
        stim.no_easyDis{curr_sub, c}(idx_excld, :)         = NaN;
        stim.no_hardDis{curr_sub, c}(idx_excld, :)         = NaN;
        perf.hitMiss{curr_sub, c}(idx_excld, :)            = NaN;

        gazeShifts     = sacc.gazeShifts{curr_sub, c};
%         gazeShifts_zen = sacc.gazeShifts_zen{curr_sub, c};
        if ~isempty(gazeShifts)

            li_excld                         = ismember(gazeShifts(:, 26), idx_excld);
%             li_excld_zen                     = ismember(gazeShifts_zen(:, 16), idx_excld);
            gazeShifts(li_excld, :)          = NaN;
%             gazeShifts_zen(li_excld, :)      = NaN;
            sacc.gazeShifts{curr_sub, c}     = gazeShifts;
%             sacc.gazeShifts_zen{curr_sub, c} = gazeShifts_zen;

        end
        clear gazeShifts gazeShifts_zen li_excld li_excld_zen

        % Calculat proportion valid trials
        exper.prop.val_trials(curr_sub, c) = 1 - numel(idx_excld) / exper.trialNo(curr_sub, c);

        % Calculate proportion trials with at least one fixated AOI
        sacc.propGs.aoiFix_mean(curr_sub, c) = ...
            sum(sacc.propGs.aoiFix{curr_sub, c}, 'omitnan') / (exper.trialNo(curr_sub, c) - numel(idx_excld));

        % Calculate time lost due to excluded trials and store # excluded trials
        exper.timeLostExcldTrials(curr_sub, c) = ...
            exper.timeLostExcldTrials(curr_sub, c) + sum(exper.cum_trialTime{curr_sub, c}(idx_excld)) / 1000;
        exper.noExcludedTrial(curr_sub, c) = numel(idx_excld);
        clear idx_excld

    end
    clear s curr_sub

end
clear c


%% Exclude subjects
% We exclude subjects based on the following criteria:
% -- For some reason, only participated in one out of two conditions
%    For this, we just check if the number of completed trials is missing
%    for one of the conditions; if it is, the conditions was not done
% -- Too little proportion trials in double-target condition with
%    correspondence between last fixated and responded on target?
% -- Too little trials with response time?

% Calculate proportion trials in which last fixated and responded on target
% corresponded as well as proportion trials in which where able to calculate
% response time
exper.prop.resp_trials       = NaN(exper.num.subNo, exper.num.condNo);
exper.prop.correspond_trials = NaN(exper.num.subNo, exper.num.condNo-1);
for c = 1:exper.num.condNo % Condition

    curr_cond = exper.num.conds(c);
    for s = 1:exper.num.subNo % Subject

        curr_sub  = exper.num.subs(s);
        idx_excld = sort(unique(exper.excl_trials{curr_sub, c}));
        no_valid  = exper.trialNo(curr_sub, c) - numel(idx_excld); % # valid trials

        % Calculat proportion trials for which we could calculate the decision time
        time_decision = sacc.time.decision{curr_sub, c};

        exper.prop.resp_trials(curr_sub, c) = sum(~isnan(time_decision)) / no_valid;
        clear time_decision

        % Calculate proportion trials in which the last fixated and the
        % responded on target corresponded
        if mod(curr_cond, 2) == 1 % Only doube-target condition

            no_correspond = sum(stim.choiceCorrespond{curr_sub, c} == 1); % # trials with correspondence

            exper.prop.correspond_trials(curr_sub) = no_correspond / no_valid;
            clear no_valid no_correspond

        end
        clear idx_excld

    end
    clear s curr_sub

    % Plot proportion trials where responded on and last fixated target correspond
%     if curr_cond == 3 % Only doube-target condition
% 
%         corr_dat = exper.prop.correspond_trials;
%         lat_dat  = vertcat(sacc.time.resp_bg{:, c});
% 
%         fig.h = figure;
%         infSampling_plt_propCorresponding(corr_dat, lat_dat, plt)
%         opt.imgname = strcat(plt.name.aggr(1:end-2), 'trial_congruency');
%         opt.size    = [45 20];
%         opt.save    = plt.save;
%         prepareFigure(fig.h, opt)
%         close; clear fig opt plt_dat lat_dat corr_dat
% 
%     end

end
clear c curr_cond

% Exclude subjects based on defined criteria
idx_excld = logical(sum(isnan(exper.trialNo), 2));

exper.events.stim_onOff(idx_excld, :)   = {[]};
sacc.time.planning(idx_excld, :)        = {[]};
sacc.time.inspection(idx_excld, :)      = {[]};
sacc.time.decision(idx_excld, :)        = {[]};
sacc.time.resp_bg(idx_excld, :)         = {[]};
sacc.time.search(idx_excld, :)          = {[]};
sacc.propGs.closest(idx_excld, :)       = {[]};
sacc.propGs.further(idx_excld, :)       = {[]};
sacc.propGs.aoiFix(idx_excld, :)        = {[]};
stim.chosenTarget(idx_excld, :)         = {[]};
stim.choiceCorrespond(idx_excld, :)     = {[]};
stim.no_easyDis(idx_excld, :)           = {[]};
stim.no_hardDis(idx_excld, :)           = {[]};
perf.hitMiss(idx_excld, :)              = {[]};
sacc.gazeShifts(idx_excld, :)           = {[]};
% sacc.gazeShifts_zen(idx_excld, :)       = {[]};
exper.trialNo(idx_excld, :)             = NaN;
exper.excl_trials(idx_excld, :)         = {[]};
perf.score.final(idx_excld, :)          = NaN;
exper.prop.val_trials(idx_excld, :)     = NaN;
exper.timeLostExcldTrials(idx_excld, :) = NaN;
exper.noExcludedTrial(idx_excld, :)     = NaN;
sacc.propGs.aoiFix_mean(idx_excld, :)   = NaN;
exper.prop.resp_trials(idx_excld, :)    = NaN;
exper.prop.correspond_trials(idx_excld) = NaN;
clear idx_excld


%% Proportion correct
perf.hitrates             = NaN(exper.num.subNo, exper.num.condNo, 3);
perf.hitrates_withDecTime = NaN(3, 2, exper.num.subNo, exper.num.condNo);
for c = 1:exper.num.condNo % Condition

    % Proportion correct for individual subjects
    for s = 1:exper.num.subNo % Subject

        curr_sub         = exper.num.subs(s);
        inp_chosenTarget = stim.chosenTarget{curr_sub, c};
        inp_hitMiss      = perf.hitMiss{curr_sub, c};
        inp_decisionTime = sacc.time.decision{curr_sub, c};
        inp_noDis        = [stim.no_easyDis{s, c} stim.no_hardDis{s, c}];

        if ~isempty(inp_chosenTarget)

            [~, perf.hitrates(curr_sub, c, 1:3), perf.hitrates_withDecTime(:, :, s, c)] = ...
                infSampling_propCorrect(inp_hitMiss, inp_chosenTarget, inp_decisionTime, inp_noDis, c);

        end
        clear curr_sub inp_chosenTarget inp_hitMiss inp_decisionTime inp_noDis

    end
    clear s

end


%% Proportion choices easy target
stim.propChoice.easy = NaN(9, exper.num.subNo, exper.num.condNo);
for c = 2:exper.num.condNo % Condition

    for s = 1:exper.num.subNo % Subject

        % Get data of subject
        curr_sub       = exper.num.subs(s);
        dat_sub_choice = stim.chosenTarget{curr_sub, c};
        dat_sub_ed     = stim.no_easyDis{curr_sub, c};

        % For each set-size, determine proportion choices easy target
        ind_ss = unique(dat_sub_ed(~isnan(dat_sub_ed)));
        no_ss  = numel(ind_ss);
        for ss = 1:no_ss

            no_trials_val  = sum(dat_sub_ed == ind_ss(ss));
            no_trials_easy = sum(dat_sub_choice == stim.identifier(1, 1) & ...
                                 dat_sub_ed == ind_ss(ss));

            stim.propChoice.easy(ss, curr_sub, c) = no_trials_easy / no_trials_val;
            clear no_trials_val no_trials_easy

        end
        clear curr_sub dat_sub_choice dat_sub_ed ind_ss no_ss ss

    end
    clear s

end
clear c


%% Proportion gaze shifts on easy set as a function of set-size
sacc.propGs.onEasy_noLock_indSs = NaN(9, exper.num.subNo, exper.num.condNo);
for c = 2:exper.num.condNo % Condition

    curr_cond = exper.num.conds(c);
    for s = 1:exper.num.subNo % Subject

        % Get data of subject and drop excluded trials
        curr_sub = exper.num.subs(s);
        dat_sub  = sacc.gazeShifts{curr_sub, c};
        if ~isempty(dat_sub)

            % Get rid of entries we do not care about
            dat_sub = dat_sub(dat_sub(:, 18) ~= stim.identifier_bg, :);  % Gaze shifts on background
            dat_sub = dat_sub(~isnan(dat_sub(:, 23)), :);                % Excluded trials & trials without choice

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

            inp_coiLab   = stim.identifier; % Category of interest
            inp_ssGroups = (0:8)';          % Analyse each set-size seperately
            inp_lock     = 2;               % Arbitrary choice, since we do not assume any timecourse for this analysis
            propGs_onEasy_noLock_indSs = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups);

            % Unpack and store data
            sacc.propGs.onEasy_noLock_indSs(:, curr_sub, c) = ...
                cell2mat(cellfun(@(x) x(:, 4, 3), propGs_onEasy_noLock_indSs, ...
                         'UniformOutput', false));
            clear inp_mat inp_coiLab inp_ssGroups inp_lock propGs_onEasy_noLock_indSs

        end
        clear curr_sub

    end
    clear s

    % Plot proportion gaze shifts on easy set against proportion choice for easy target
%     if curr_cond == 3 % Double-target condition
% 
%         % Unpack data
%         prop_choiceEasy = stim.propChoice.easy(:, :, c);
%         prop_saccEasy   = sacc.propGs.onEasy_noLock_indSs(:, :, c);
%         dat             = [prop_choiceEasy(:) prop_saccEasy(:)];
% 
%         infSampling_plt_corrChoiceEasyPropGsEasy(dat, plt)
%         clear dat prop_choiceEasy prop_saccEasy
% 
%     end

end
clear c curr_cond


%% Timecourse proportion gaze shifts on stimulus in trial
sacc.propGs.onChosen_trialBegin  = cell(exper.num.subNo, exper.num.condNo);
sacc.propGs.onEasy_trialBegin    = cell(exper.num.subNo, exper.num.condNo);
sacc.propGs.onSmaller_trialBegin = cell(exper.num.subNo, exper.num.condNo);
sacc.propGs.onCloser_trialBegin  = cell(exper.num.subNo, exper.num.condNo);
for c = 2:exper.num.condNo % Condition; only double-target

    for s = 1:exper.num.subNo % Subject

        % Get data of subject and drop excluded trials
        curr_sub = exper.num.subs(s);
        dat_sub  = sacc.gazeShifts{curr_sub, c};
        if ~isempty(dat_sub)

            % Get rid of entries we do not care about
            dat_sub       = dat_sub(dat_sub(:, 18) ~= stim.identifier_bg, :); % Saccades on background
            dat_sub       = dat_sub(~isnan(dat_sub(:, 23)), :);               % Excluded trials & trials without choice
            dat_sub_noMed = dat_sub(dat_sub(:, 22) ~= 4, :);                  % Excluded trials with equal # easy/difficult distractors (onyl for proportion smaller set)

            % Get stimulus category of interest, for which we want to
            % analyse proportion gaze shift that landed on it
            li_gsOnChosenSet   = any(dat_sub(:, 18) == stim.identifier(:, dat_sub(:, 23))', 2);                                                         % Set of chosen target
            li_gsOnEasySet     = any(dat_sub(:, 18) == stim.identifier(:, 1)', 2);                                                                      % Easy set
            li_gsOnClosestStim = dat_sub(:, 21) == 1;                                                                                                   % Closest stimulus
            li_gsOnSmallerSet  = (any(dat_sub_noMed(:, 18) == stim.identifier(:, 1)', 2) & dat_sub_noMed(:, 22) >= 0 & dat_sub_noMed(:, 22) <= 3) | ... % Smaller set
                                 (any(dat_sub_noMed(:, 18) == stim.identifier(:, 2)', 2) & dat_sub_noMed(:, 22) >= 5 & dat_sub_noMed(:, 22) <= 8);

            % Timecourse of proportion saccades to chosen set
            no_gs   = size(dat_sub, 1);
            inp_mat = [dat_sub(:, 24) ...   Timelock
                       NaN(no_gs, 1) ...    Legacy column
                       dat_sub(:, 18) ...   AOI identifier
                       li_gsOnChosenSet ... Stimulus category of interest
                       dat_sub(:, 22) ...   Number easy distractors
                       dat_sub(:, 7) ...    Timestamp gaze shift offset
                       dat_sub(:, 26) ...   Trial number
                       dat_sub(:, 23)];   % Target chosen in trial
            clear dat_sub no_gs

            inp_coiLab   = stim.identifier; % Category of interest == chosen target
            inp_ssGroups = (0:8)';          % Analyse over all set-sizes
            inp_lock     = 2;               % Locked to trial beginning
            sacc.propGs.onChosen_trialBegin{curr_sub, c} = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups);

            % Timecourse of proportion gaze shifts to easy set
            inp_mat(:, 4) = li_gsOnEasySet;
            sacc.propGs.onEasy_trialBegin{curr_sub, c} = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups);

            % Timecourse of proportion gaze shifts to closer stimulus
            inp_mat(:, 4) = li_gsOnClosestStim;
            sacc.propGs.onCloser_trialBegin{curr_sub, c} = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups);
            clear li_gsOnChosenSet li_gsOnEasySet li_gsOnClosestStim inp_mat

            % Get timecourse of proportion gaze shifts on smaller set
            no_gs   = size(dat_sub_noMed, 1);
            inp_mat = [dat_sub_noMed(:, 24) ...   Timelock
                       NaN(no_gs, 1) ...          Legacy column
                       dat_sub_noMed(:, 18) ...   AOI identifier
                       li_gsOnSmallerSet ...      Stimulus category of interest
                       dat_sub_noMed(:, 22) ...   Number easy distractors
                       dat_sub_noMed(:, 7) ...    Timestamp gaze shift offset
                       dat_sub_noMed(:, 26) ...   Trial number
                       dat_sub_noMed(:, 23)];   % Target chosen in trial

            sacc.propGs.onSmaller_trialBegin{curr_sub, c} = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups);
            clear li_gsOnSmallerSet inp_coiLab inp_ssGroups inp_lock dat_sub_noMed no_gs inp_mat

        end
        clear curr_sub dat_sub 

    end
    clear s

end
clear c


%% Planning, inspection and decision times
sacc.time.mean.planning   = NaN(exper.num.subNo, exper.num.condNo, 3);
sacc.time.mean.inspection = NaN(exper.num.subNo, exper.num.condNo, 3);
sacc.time.mean.decision   = NaN(exper.num.subNo, exper.num.condNo, 3);
sacc.time.mean.non_search = NaN(exper.num.subNo, exper.num.condNo, 3);
for c = 1:exper.num.condNo % Condition

    for s = 1:exper.num.subNo % Subject

        curr_sub = exper.num.subs(s);

        dat_chosenTarg_sub = stim.chosenTarget{curr_sub, c};
        dat_noDis_sub      = [stim.no_easyDis{curr_sub, c} stim.no_hardDis{curr_sub, c}];
        dat_planTime_sub   = sacc.time.planning{curr_sub, c};
        dat_inspTime_sub   = sacc.time.inspection{curr_sub, c};
        dat_decTime_sub    = sacc.time.decision{curr_sub, c};
        if ~isempty(dat_inspTime_sub)

            setSizes = unique(dat_noDis_sub(~isnan(dat_noDis_sub(:, 1)), 1));
            NOSS     = numel(setSizes);
            for t = 1:3 % Target difficulty

                temp = NaN(4, NOSS);
                for ss = 1:NOSS % Set size

                    switch t

                        case 1
                            switch c % Single-/double-target condtion
                                case 1 % Single-target: trials where easy/difficult target was shown with given number of distractors
                                    li_trials = any(dat_noDis_sub == setSizes(ss), 2);

                                case 2 % Double-target: trials where easy/difficult was chosen with given number of same colored distractors
                                    li_trials = dat_noDis_sub(:, 1) == setSizes(ss);

                            end

                        otherwise
                            switch c % Single-/double-target condtion
                                case 1
                                    li_trials = dat_noDis_sub(:, t-1) == setSizes(ss) & dat_chosenTarg_sub == t-1;

                                case 2
                                    li_trials = dat_noDis_sub(:, 1) == setSizes(ss) & dat_chosenTarg_sub == t-1;

                            end

                    end
                    temp(:, ss) = [mean(dat_planTime_sub(li_trials), 'omitnan'); ...
                                   mean(dat_inspTime_sub(li_trials), 'omitnan'); ...
                                   mean(dat_decTime_sub(li_trials), 'omitnan'); ...
                                   mean(dat_planTime_sub(li_trials) + dat_decTime_sub(li_trials), 'omitnan')];
                    clear li_trials

                end
                sacc.time.mean.planning(curr_sub, c, t)   = mean(temp(1, :), 2, 'omitnan');
                sacc.time.mean.inspection(curr_sub, c, t) = mean(temp(2, :), 2, 'omitnan');
                sacc.time.mean.decision(curr_sub, c, t)   = mean(temp(3, :), 2, 'omitnan');
                sacc.time.mean.non_search(curr_sub, c, t) = mean(temp(4, :), 2, 'omitnan');
                clear temp ss

            end
            clear t NOSS

        end
        clear curr_sub dat_noDis_sub dat_planTime_sub dat_inspTime_sub dat_decTime_sub dat_planTime_sub dat_chosenTarg_sub setSizes

    end
    clear s

end
clear c


%% How much time participants spent searching for targets
sacc.time.search_reg_coeff = NaN(exper.num.subNo, 2, exper.num.condNo);
sacc.time.search_confInt   = NaN(2, 2, exper.num.subNo, exper.num.condNo);
sacc.time.search_ss        = NaN(exper.num.subNo, 9, exper.num.condNo);
for c = 1:exper.num.condNo % Condition

    for s = 1:exper.num.subNo % Subject

        curr_sub   = exper.num.subs(s);
        searchTime = sacc.time.search{curr_sub, c};
        if ~isempty(searchTime)

            searchTime = sacc.time.search{curr_sub, c}(:, 4);
            noDis_sub  = [stim.no_easyDis{curr_sub, c} stim.no_hardDis{curr_sub, c}];
            no_ss      = unique(noDis_sub(~isnan(noDis_sub(:, 1)), 1));
            for ss = 1:numel(no_ss) % Set size

                switch c

                    case 1
                        li_trials = any(noDis_sub == no_ss(ss), 2);

                    case 2
                        li_trials = noDis_sub(:, 1) == no_ss(ss);

                end

                sacc.time.search_ss(curr_sub, ss, c) = mean(searchTime(li_trials), 'omitnan');
                clear li_trials

            end
            clear no_ss ss noDis_sub

            % Regression over mean inspection time for different set sizes
            reg_predictor = (0:8)';
            reg_criterion = sacc.time.search_ss(curr_sub, :, c)';

            [sacc.time.search_reg_coeff(curr_sub, :, c), sacc.time.search_confInt(:, :, curr_sub, c)] = ...
                regress(reg_criterion, [ones(numel(reg_predictor), 1) reg_predictor]);
            clear reg_predictor reg_criterion

        end
        clear curr_sub searchTime

    end
    clear s

end
clear c


%% Proportion gaze shifts to chosen/not-chosen targets/distractors and background
% All gaze shifts are considered; proportions are calculaed over all gaze
% shifts, without separating by set size first
sacc.propGs.onAOI    = NaN(exper.num.subNo, 5, exper.num.condNo);
sacc.propGs.onAOI_ss = NaN(exper.num.subNo, 9, 3, exper.num.condNo);
for c = 1:exper.num.condNo % Condition

    for s = 1:exper.num.subNo % Subject

        curr_sub = exper.num.subs(s);
        gs_sub   = sacc.gazeShifts{curr_sub, c};
        if ~isempty(gs_sub)

            fixatedStim                       = gs_sub(:, 18);
            li_validGs                        = ~isnan(fixatedStim);
            id_chosenTarget                   = gs_sub(:, 23);
            noValidGs                         = sum(~isnan(id_chosenTarget)); % Use this variable to account for trials without choices
            li_easyChosen                     = id_chosenTarget == stim.identifier(1, 1);
            li_diffChosen                     = id_chosenTarget == stim.identifier(1, 2);
            id_nonChosenTarget                = NaN(numel(fixatedStim), 1);
            id_nonChosenTarget(li_easyChosen) = stim.identifier(1, 2);
            id_nonChosenTarget(li_diffChosen) = stim.identifier(1, 1);
            clear li_easyChosen li_diffChosen

            li_gsOnChosenTarget    = fixatedStim(li_validGs) == stim.identifier(1, id_chosenTarget(li_validGs))';
            li_gsOnNonChosenTarget = fixatedStim(li_validGs) == stim.identifier(1, id_nonChosenTarget(li_validGs))';
            li_gsOnChosenDist      = fixatedStim(li_validGs) == stim.identifier(2, id_chosenTarget(li_validGs))';
            li_gsOnNonChosenDist   = fixatedStim(li_validGs) == stim.identifier(2, id_nonChosenTarget(li_validGs))';
            li_gsOnBackground      = fixatedStim(li_validGs) == stim.identifier_bg;
            clear li_validGs

            sacc.propGs.onAOI(s, :, c) = [sum(li_gsOnChosenTarget)    / noValidGs ...
                                          sum(li_gsOnNonChosenTarget) / noValidGs ...
                                          sum(li_gsOnChosenDist)      / noValidGs ...
                                          sum(li_gsOnNonChosenDist)   / noValidGs ...
                                          sum(li_gsOnBackground)      / noValidGs];
            clear noValidGs li_gsOnChosenTarget li_gsOnNonChosenTarget li_gsOnChosenDist li_gsOnNonChosenDist li_gsOnBackground

            % Calculate proportion fixations on chosen/not-chosen set, as a function of set size
            setSizes = unique(gs_sub(:, 22));
            setSizes = setSizes(~isnan(setSizes));
            for ss = 1:numel(setSizes) % Set size

                % In the single-target condition, # difficult distractors
                % have to be considered as well, whereas in the
                % double-target condition, those trials are naturally
                % included in the # easy distractors
                switch c

                    case 1
                        li_ss = any(gs_sub(:, [22 28]) == setSizes(ss), 2);

                    case 2
                        li_ss = gs_sub(:, 22) == setSizes(ss);

                end
                fixatedStim_ss        = fixatedStim(li_ss);
                id_chosenTarget_ss    = id_chosenTarget(li_ss);
                id_nonChosenTarget_ss = id_nonChosenTarget(li_ss);

                li_gsOnChosenTarget    = any(fixatedStim_ss == stim.identifier(:, id_chosenTarget_ss)', 2);
                li_gsOnNonChosenTarget = any(fixatedStim_ss == stim.identifier(:, id_nonChosenTarget_ss)', 2);
                li_gsOnBackground      = fixatedStim_ss == stim.identifier_bg;
                noValidGs              = sum(li_ss);
                clear li_ss fixatedStim_ss id_chosenTarget_ss id_nonChosenTarget_ss

                sacc.propGs.onAOI_ss(s, ss, :, c) = [sum(li_gsOnChosenTarget)    / noValidGs ...
                                                     sum(li_gsOnNonChosenTarget) / noValidGs ...
                                                     sum(li_gsOnBackground)      / noValidGs];
                clear li_gsOnChosenTarget li_gsOnNonChosenTarget li_gsOnChosenDist li_gsOnNonChosenDist li_gsOnBackground noValidGs

            end
            clear fixatedStim id_chosenTarget id_nonChosenTarget setSizes ss

        end
        clear curr_sub gs_sub

    end
    clear s

end
clear c


%% Proportion gaze shifts to chosen/not-chosen stimuli, as a function of set size
% Only gaze shifts to distractors as well as the last gaze shift to a
% target (if it was not followed by another gaze shift to a distractor) are
% counted; proportions are calculaed seperately for each set size and
% across alls gaze shifts, without separating for set sizes
sacc.propGs.onAOI_modelComparision_chosenNot    = NaN(exper.num.subNo, 2, exper.num.condNo); % Proportion fixations on chosen/not-chosen set
sacc.propGs.onAOI_modelComparision_chosenNot_ss = NaN(exper.num.subNo, 9, exper.num.condNo); % Proportion fixations on chosen/not-chosen set, seperate for set-sizes
sacc.propGs.onAOI_modelComparision_easyDiff     = NaN(exper.num.subNo, 2, exper.num.condNo); % Proportion fixations on easy/difficult set
sacc.propGs.onAOI_modelComparision_easyDiff_ss  = NaN(exper.num.subNo, 9, exper.num.condNo); % Proportion fixations on easy/difficult set, seperate for set-sizes
for c = 2:exper.num.condNo % Condition

    for s = 1:exper.num.subNo % Subject

        curr_sub = exper.num.subs(s);
        gs_sub   = sacc.gazeShifts{curr_sub, c};
        if ~isempty(gs_sub)

            % Extract unique fixations of elements
            % Removes multiple fixations of same element in trials as well
            % as fixations on targets, except for the last fixation in a
            % trial (if this one went to a target)
            gs_sub2 = [];
            for t = 1:exper.trialNo(s, c) % Trial

                dat_trial                        = gs_sub(gs_sub(:, 26) == t, :);
                li_targWhileSearch               = any(dat_trial(1:end-1, 18) == stim.identifier(1, :), 2); % Target fixation while searching
                dat_trial(li_targWhileSearch, :) = [];
                clear li_targWhileSearch

                [~, ia, ~] = unique(dat_trial(:, 17));

                gs_sub2 = [gs_sub2; sortrows(dat_trial(ia, :), 1)];
                clear dat_trial ia

            end
            gs_sub = gs_sub2;
            clear t gs_sub2

            % Calculate proportion fixations on chosen/not-chosen set
            fixatedStim                       = gs_sub(:, 18);
            id_chosenTarget                   = gs_sub(:, 23);
            li_easyChosen                     = id_chosenTarget == stim.identifier(1, 1);
            li_diffChosen                     = id_chosenTarget == stim.identifier(1, 2);
            id_nonChosenTarget                = NaN(numel(fixatedStim), 1);
            id_nonChosenTarget(li_easyChosen) = stim.identifier(1, 2);
            id_nonChosenTarget(li_diffChosen) = stim.identifier(1, 1);
            clear li_easyChosen li_diffChosen

            li_onDis           = any(fixatedStim == stim.identifier(:)', 2);
            li_onDis_easy      = any(fixatedStim(li_onDis) == stim.identifier(:, 1)', 2);
            li_onDis_diff      = any(fixatedStim(li_onDis) == stim.identifier(:, 2)', 2);
            li_onDis_chosen    = any(fixatedStim(li_onDis) == stim.identifier(:, id_chosenTarget(li_onDis))', 2);
            li_onDis_nonChosen = any(fixatedStim(li_onDis) == stim.identifier(:, id_nonChosenTarget(li_onDis))', 2);
            noValidGs          = sum(~isnan(id_chosenTarget(li_onDis)));

            sacc.propGs.onAOI_modelComparision_chosenNot(s, :, c) = [sum(li_onDis_chosen) / noValidGs ...
                                                                     sum(li_onDis_nonChosen) / noValidGs];
            sacc.propGs.onAOI_modelComparision_easyDiff(s, :, c)  = [sum(li_onDis_easy) / noValidGs ...
                                                                     sum(li_onDis_diff) / noValidGs];
            clear li_onDis li_onDis_easy li_onDis_diff li_onDis_chosen li_onDis_nonChosen noValidGs

            % Calculate proportion fixations on chosen/not-chosen set, as a function of set size
            setSizes = unique(gs_sub(:, 22));
            setSizes = setSizes(~isnan(setSizes));
            for ss = 1:numel(setSizes) % Set size

                li_ss                 = gs_sub(:, 22) == setSizes(ss);
                fixatedStim_ss        = fixatedStim(li_ss);
                id_chosenTarget_ss    = id_chosenTarget(li_ss);
                id_nonChosenTarget_ss = id_nonChosenTarget(li_ss);

                li_onDis           = any(fixatedStim_ss == stim.identifier(:)', 2);
                li_onDis_easy      = any(fixatedStim_ss(li_onDis) == stim.identifier(:, 1)', 2);
                li_onDis_diff      = any(fixatedStim_ss(li_onDis) == stim.identifier(:, 2)', 2);
                li_onDis_chosen    = any(fixatedStim_ss(li_onDis) == stim.identifier(:, id_chosenTarget_ss(li_onDis))', 2);
                li_onDis_nonChosen = any(fixatedStim_ss(li_onDis) == stim.identifier(:, id_nonChosenTarget_ss(li_onDis))', 2);
                noValidGs          = sum(~isnan(id_chosenTarget_ss(li_onDis)));

                sacc.propGs.onAOI_modelComparision_chosenNot_ss(s, ss, c) = sum(li_onDis_chosen) / noValidGs;
                sacc.propGs.onAOI_modelComparision_easyDiff_ss(s, ss, c)  = sum(li_onDis_easy) / noValidGs;
                clear li_ss fixatedStim_ss id_chosenTarget_ss id_nonChosenTarget_ss li_onDis li_onDis_easy li_onDis_diff li_onDis_chosen li_onDis_nonChosen noValidGs

            end
            clear fixatedStim id_chosenTarget id_nonChosenTarget setSizes ss

        end
        clear curr_sub gs_sub

    end
    clear s

end
clear c


%% Export data for model
% Model scripts are build around getting data from exported .txt files and
% fitting the model to the imported data. To make things easier, I will
% keep this workflow, instead of "properly" implementing the model scripts
% into my framework
container_dat_mod   = NaN(exper.num.subNo, 100, 2);
container_dat_label = infSampling_colNames;
dat_filenames       = {[exper.name.export{exper.num.conds(1)-1, 1}, '.txt'],  [exper.name.export{exper.num.conds(1)-1, 2}, '.txt']; ...
                       [exper.name.export{exper.num.conds(1)-1, 1}, '.xlsx'], [exper.name.export{exper.num.conds(1)-1, 2}, '.xlsx']};
for c = 1:exper.num.condNo % Condition

    % Gather data to export
    container_dat = [exper.num.subs ...                                         1:     Subject numbers
                     reshape(perf.hitrates(:, c, :), exper.num.subNo, 3, 1) ...        Proportion correct (overall, easy, difficult)
                     NaN(exper.num.subNo, 60) ...                               5:64:  Placeholder for legacy columns
                     sacc.time.mean.inspection(:, c, 1) ...                     65:    Overall mean inspection time per item
                     sacc.time.mean.non_search(:, c, 1) ...                            Overall mean non-search time
                     sacc.time.mean.planning(:, c, 1) ...                              Overall mean planning time
                     sacc.time.mean.decision(:, c, 1) ...                              Overall mean decision time
                     sacc.time.mean.inspection(:, c, 2) ...                            Mean search time easy target chosen
                     sacc.time.mean.non_search(:, c, 2) ...                     70:    Mean non-search time easy target chosen
                     sacc.time.mean.decision(:, c, 2) ...                              Mean decision time easy target chosen
                     sacc.time.mean.planning(:, c, 2) ...                              Mean planning time easy target chosen
                     sacc.time.mean.inspection(:, c, 3) ...                            Mean inspection time per item difficult target chosen
                     sacc.time.mean.non_search(:, c, 3) ...                            Mean non-search time difficult target chosen
                     sacc.time.mean.decision(:, c, 3) ...                       75:    Mean response time difficult target chosen
                     sacc.time.mean.planning(:, c, 3) ...                              Mean fixation time difficult target chosen
                     NaN(exper.num.subNo, 2) ...                                77:78: Placeholder for legacy columns
                     stim.propChoice.easy(:, :, c)' ...                         79:87: Proportion choices easy target as a function of set-size
                     1-stim.propChoice.easy(:, :, c)' ...                       88:96: Proportion choices difficult target as a function of set-size
                     exper.trialNo(:, c) ...                                           # solved trials
                     exper.timeLostExcldTrials(:, c) ...                               time lost due to excluded trials
                     exper.noExcludedTrial(:, c) ...                                   # excluded trials
                     perf.score.final(:, c)];                                 % 100: accumulated reward
    container_dat_mod(:, :, c) = container_dat;

    % Export data
    if exper.flag.export == 1

        % Define paths
        savePath_txt = strcat(exper.name.analysis, '/_model/', dat_filenames{1, c});
        savePath_xls = strcat(exper.name.analysis, '/_model/', dat_filenames{2, c});

        % Delete old files to prevent weird bug that might occur due to
        % overwriting existing files
        delete(savePath_txt, savePath_xls);

        % Save data as .txt and .xls
        writematrix(container_dat, savePath_txt);
        container_dat_xls = num2cell(container_dat);
        container_dat_xls(isnan(container_dat)) = {'NaN'};
        dat_table = array2table(container_dat_xls, ...
                                'VariableNames', container_dat_label');
        writetable(dat_table, savePath_xls)
        clear savePath_txt savePath_xls

    end
    clear container_dat container_dat_xls dat_table

end
clear c container_dat_label dat_filenames


%% Fit model with perfect fixation distribution
cd(strcat(exper.name.analysis, '/_model'))
model_io = [];
model_io.containerDat = container_dat_mod; % Get data from .xls files
model_io = get_params(model_io);
model_io = read_data(model_io);
model_io = fit_model(model_io); % Fit model and plot results
model_io = fit_regression(model_io); % Fit regression and plot results
clear container_dat_mod


%% Fit probabilistic model
cd('/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual/3_analysis/_model/_recursiveModel_standalone');
% infSampling_generateLUT([(1:9)' (9:-1:1)'], [0 2], 4, 1)
% model = infSampling_model_main(stim, sacc, model_io, perf, plt);
load('modelResults_propChoices_fixChosen.mat');


%% Statistics for paper 
% Some general descriptive stuff
clc; disp([round(mean(perf.score.final, 'omitnan'), 2); ...                                                                       Final scores at end of conditions
           round(std(perf.score.final,  'omitnan'), 2); ...
           min(perf.score.final); ...
           max(perf.score.final)]);
clc; disp([round(mean(sum(perf.score.final, 2), 'omitnan'), 2); ...
           round(std(sum(perf.score.final, 2),  'omitnan'), 2); ...
           min(sum(perf.score.final, 2)); ...
           max(sum(perf.score.final, 2))]);
clc; matlab_pairedTtest(perf.score.final(:, 1), perf.score.final(:, 2), 2)
clc; disp([round(mean(exper.prop.val_trials, 'omitnan') .* 100, 2); ...                                                           Percentage valid trials
           round(min(exper.prop.val_trials) .* 100, 2); ...
           round(max(exper.prop.val_trials) .* 100, 2)]);
clc; disp([round(mean(exper.prop.resp_trials, 'omitnan') .* 100, 2); ...                                                          Percentage trials where decision time could be calculated
           round(min(exper.prop.resp_trials) .* 100, 2); ...
           round(max(exper.prop.resp_trials) .* 100, 2)]);
clc; disp([round(mean(exper.trialNo, 'omitnan')); ...                                                                             Completed trials
           round(std(exper.trialNo,  'omitnan')); ...
           min(exper.trialNo); ...
           max(exper.trialNo)]);
clc; disp(round(mean(sacc.propGs.aoiFix_mean, 'omitnan') .* 100, 2));                                                           % Percentage trials with at least one fixated AOI
clc; round(squeeze(mean(sacc.time.search_ss(2, :, :), 2))')                                                                  % Search time of extreme and remaining subjects
clc; disp([round(squeeze(mean(mean(sacc.time.search_ss, 2), 'omitnan'))'); ...  
           round(squeeze(mean(mean(sacc.time.search_ss, 2), 'omitnan'))'-ci_mean(squeeze(mean(sacc.time.search_ss, 2)))); ... 
           round(squeeze(mean(mean(sacc.time.search_ss, 2), 'omitnan'))'+ci_mean(squeeze(mean(sacc.time.search_ss, 2))))])

% Figure 3
temp = squeeze(mean(sacc.propGs.onAOI_ss(:, :, :, 1), 2, 'omitnan'));
clc; disp([round(mean(temp, 1, 'omitnan'), 2); ...                                                                                 3A: Proportion gaze shifts on AOIs
           round(mean(temp, 1, 'omitnan')-ci_mean(temp), 2); ...
           round(mean(temp, 1, 'omitnan')+ci_mean(temp), 2)])
clear temp
clc; disp([round(mean(sacc.time.search_reg_coeff(:, :, 1), 'omitnan')); ...                                                     3B: Search time over set sizes
           round(mean(sacc.time.search_reg_coeff(:, :, 1), 'omitnan')-ci_mean(sacc.time.search_reg_coeff(:, :, 1))); ... 
           round(mean(sacc.time.search_reg_coeff(:, :, 1), 'omitnan')+ci_mean(sacc.time.search_reg_coeff(:, :, 1)))])
clc; matlab_oneSampleTtest(sacc.time.search_reg_coeff(:, 2, 1), 0);
clc; disp([round(mean(sacc.time.mean.inspection(:, 1, 1), 'omitnan')); ...
           round(mean(sacc.time.mean.inspection(:, 1, 1), 'omitnan')-ci_mean(sacc.time.mean.inspection(:, 1, 1))); ... 
           round(mean(sacc.time.mean.inspection(:, 1, 1), 'omitnan')+ci_mean(sacc.time.mean.inspection(:, 1, 1)))])
clc; matlab_pairedTtest(perf.hitrates(:, 1, 2),             perf.hitrates(:, 1, 3), 2)                                              % 3C: Discrimination difficulty
clc; matlab_pairedTtest(sacc.time.mean.planning(:, 1, 2),   sacc.time.mean.planning(:, 1, 3), 0);                                   % 3D: Planning time
clc; matlab_pairedTtest(sacc.time.mean.inspection(:, 1, 2), sacc.time.mean.inspection(:, 1, 3), 0);                                 % 3E: Inspection time
clc; matlab_pairedTtest(sacc.time.mean.decision(:, 1, 2),   sacc.time.mean.decision(:, 1, 3), 0);                                   % 3F: Decision time

% Figure 4
% Intercepts and slopes of regression fit
clc; matlab_oneSampleTtest(model_io.reg.fit(:, 1), 2); % Intercepts
clc; matlab_oneSampleTtest(model_io.reg.fit(:, 2), 2); % Slopes

% Figure 5
% Proportion gaze shifts on different stimuli over the course of trials
inp_minSub = exper.avg.minSub;
inp_dat    = [sacc.propGs.onChosen_trialBegin(:, 2) ...
              sacc.propGs.onEasy_trialBegin(:, 2) ...
              sacc.propGs.onSmaller_trialBegin(:, 2) ...
              sacc.propGs.onCloser_trialBegin(:, 2)];
% single_subjects = infSampling_avgPropSacc(inp_dat, inp_minSub);
single_subjects = squeeze(mean(infSampling_avgPropSacc(inp_dat, inp_minSub), 3, 'omitnan'));
clear inp_minSub

clc; matlab_pairedTtest(single_subjects(:, 1, 1), single_subjects(:, 2, 1), 2) % 5A: Proportions to chosen/not-chosen sets for first two gaze shifts
clc; matlab_pairedTtest(single_subjects(:, 1, 2), single_subjects(:, 2, 2), 2) % 5B: Proportions to easy/difficult sets for first two gaze shifts
clc; matlab_pairedTtest(single_subjects(:, 1, 3), single_subjects(:, 2, 3), 2) % 5C: Proportion to smaller/larger sets for first two gaze shifts
clc; matlab_pairedTtest(single_subjects(:, 1, 4), single_subjects(:, 2, 4), 2) % 5D: Proportions to closet/more distant sets for first two gaze shifts
clear single_subjects

% Figure 6
% Model results
clc; matlab_pairedTtest(model_io.data.double.perf, model_io.model.perf_perfect(:, 3), 2)              % 6A: Empirical vs. maximum gain
[r, p, rl, ru] = corrcoef(model_io.data.double.perf, ...
                          model_io.model.perf_perfect(:, 3), 'Rows', 'complete');
clc; disp(round([r(1, 2), rl(1, 2), ru(1, 2), p(1, 2)], 2));
clc; matlab_pairedTtest(model.freeParameter{2}(:, 1), model.freeParameter{2}(:, 2), 2)                % 6B: Free parameters distributions
clc; matlab_pairedTtest(model_io.data.double.perf, model.performance(:, 2), 2)                        % 6C: Empirical vs. stochastic model gain
[r, p, rl, ru] = corrcoef(model_io.data.double.perf, model.performance(:, 2), 'Rows', 'complete');
clc; disp(round([r(1, 2), rl(1, 2), ru(1, 2), p(1, 2)], 2));
clc; matlab_pairedTtest(mean(sacc.propGs.onAOI_modelComparision_chosenNot_ss(:, :, 2), 2), ...       6E: Empirical vs. predicted proportion gaze shifts on chosen set
                        mean(model.propFixChosen(:, :, 2), 2), 2)
[r, p, rl, ru] = corrcoef(mean(sacc.propGs.onAOI_modelComparision_chosenNot_ss(:, :, 2), 2), ...
                          mean(model.propFixChosen(:, :, 2), 2), 'Rows', 'complete');
clc; disp(round([r(1, 2), rl(1, 2), ru(1, 2), p(1, 2)], 2));

% % Figure S4
% % Latencies of first gaze shifts to different stimuli
% clc; matlab_pairedTtest(sacc.lat.firstGs_chosenSet(:, 1, 2),   sacc.lat.firstGs_chosenSet(:, 2, 2))   % S4A: Latencies to chosen/not-chosen set
% clc; matlab_pairedTtest(sacc.lat.firstGs_easySet(:, 1, 2),     sacc.lat.firstGs_easySet(:, 2, 2))     % S4A: Latencies to easy/difficult set
% clc; matlab_pairedTtest(sacc.lat.firstGs_smallerSet(:, 1, 2),  sacc.lat.firstGs_smallerSet(:, 2, 2))  % S4A: Latencies to smaller/larger set
% clc; matlab_pairedTtest(sacc.lat.firstGs_closestStim(:, 1, 2), sacc.lat.firstGs_closestStim(:, 2, 2)) % S4A: Latencies to closest/more distant stimulus


%% Create plots for paper
% Figure 3
% Proportion gaze shifts on different AOIs, search time as a function of
% distractor number, perceptual performance and temporal aspects of search
% behavior (planning-, search- and decision-time) in single-target
% condition
inp_searchTime  = [(0:8)' sacc.time.search_ss(:, :, 1)'];
inp_propGsOnAoi = [(1:3)' squeeze(mean(sacc.propGs.onAOI_ss(:, :, :, 1), 2, 'omitnan'))'];
inp_perf        = cat(3, ...
                      [perf.hitrates(:, 1, 2)             perf.hitrates(:, 1, 3)], ...
                      [sacc.time.mean.planning(:, 1, 2)   sacc.time.mean.planning(:, 1, 3)], ...
                      [sacc.time.mean.inspection(:, 1, 2) sacc.time.mean.inspection(:, 1, 3)], ...
                      [sacc.time.mean.decision(:, 1, 2)   sacc.time.mean.decision(:, 1, 3)]);
inp_pltName     = strcat(plt.name.aggr(1:end-14), 'figure3');

infSampling_plt_fig3(inp_searchTime, NaN, NaN, inp_propGsOnAoi, inp_perf, inp_pltName, plt)
clear inp_dat_var inp_dat_reg inp_dat_reg_long inp_mod_reg inp_dat_gs inp_pltName

% Figure 4
% Proportion choices easy target for representative participants from the
% double-target condition and slopes/intercepts of regressions, fitted to
% decision-curves
emp_prop_choices_easy     = stim.propChoice.easy(:, :, 2);
emp_prop_choices_easy_fit = cat(3, model_io.reg.xn-1, model_io.reg.yn);
emp_slopesIntercepts      = model_io.reg.fit;

infSampling_plt_fig4(emp_prop_choices_easy, emp_prop_choices_easy_fit, emp_slopesIntercepts, plt)
clear prop_choices_easy prop_choices_easy_fit slopesIntercepts

% Figure 5
% Proportion gaze shifts on different stimuli over the course of trials
inp_minSub = exper.avg.minSub;
inp_dat    = [sacc.propGs.onChosen_trialBegin(:, 2) ...
              sacc.propGs.onEasy_trialBegin(:, 2) ...
              sacc.propGs.onSmaller_trialBegin(:, 2) ...
              sacc.propGs.onCloser_trialBegin(:, 2)];
inp_single = squeeze(mean(infSampling_avgPropSacc(inp_dat, inp_minSub), 3, 'omitnan'));
infSampling_plt_fig5(inp_single, plt)
clear inp_mean inp_single

% Figure 6
% Results of model fitting
inp_emp_propChoicesEasy = stim.propChoice.easy(:, :, 2)';
inp_emp_propGsChosen    = sacc.propGs.onAOI_modelComparision_chosenNot_ss(:, :, 2);
inp_emp_perf            = model_io.data.double.perf;

infSampling_plt_fig6(inp_emp_propChoicesEasy, inp_emp_propGsChosen, inp_emp_perf, ...
                     model, model_io.model, plt)

% Supplementary figure 1
% Proportion gaze shifts on different AOIs, search time as a function of
% distractor number, perceptual performance and temporal aspects of search
% behavior (planning-, search- and decision-time) in single-target
% condition
inp_dat_perf = cat(3, ...
                   [perf.hitrates(:, 2, 2)             perf.hitrates(:, 2, 3)], ...
                   [sacc.time.mean.planning(:, 2, 2)   sacc.time.mean.planning(:, 2, 3)], ...
                   [sacc.time.mean.inspection(:, 2, 2) sacc.time.mean.inspection(:, 2, 3)], ...
                   [sacc.time.mean.decision(:, 2, 2)   sacc.time.mean.decision(:, 2, 3)]);
inp_dat_reg  = [(0:8)' sacc.time.search_ss(:, :, 2)'];
inp_dat_gs   = [(1:3)' squeeze(mean(sacc.propGs.onAOI_ss(:, :, :, 2), 2, 'omitnan'))'];
inp_pltName  = strcat(plt.name.aggr(1:end-14), 'figureSupp1');

infSampling_plt_fig3(inp_dat_reg, NaN, NaN, inp_dat_gs, inp_dat_perf, inp_pltName, plt)
clear inp_dat_var inp_dat_reg inp_dat_reg_long inp_mod_reg inp_dat_gs inp_pltName

% Supplementary figure 2
% Proportion choices easy target of individual subjects in double-target condition
prop_choices_easy      = stim.propChoice.easy(:, :, 2);
prop_choices_easy_fit  = cat(3, model_io.reg.xn-1, model_io.reg.yn);
prop_choices_easy_pred = model.propChoicesEasy(:, :, 2)';

infSampling_plt_figSupp2(prop_choices_easy, prop_choices_easy_fit, prop_choices_easy_pred, plt)
clear prop_choices_easy prop_choices_easy_fit prop_choices_easy_pred

% Supplementary figure 3
% Predicted proportions gaze shifts on chosen set
propFix_pred = model.propFixChosen(:, :, 2);
propFix_emp  = sacc.propGs.onAOI_modelComparision_chosenNot_ss(:, :, 2);

infSampling_plt_figSupp3(propFix_pred, propFix_emp, plt)
clear propFix_pred propFix_emp

% % Supplementary figure X
% % Latencies of first gaze shifts to different stimuli
% inp_dat = cat(3, ...
%               sacc.lat.firstGs_chosenSet(:, :, 2), ...
%               sacc.lat.firstGs_easySet(:, :, 2), ...
%               sacc.lat.firstGs_smallerSet(:, :, 2), ...
%               sacc.lat.firstGs_closestStim(:, :, 2));
% 
% infSampling_plt_figSuppThree(inp_dat, plt)
% clear inp_dat