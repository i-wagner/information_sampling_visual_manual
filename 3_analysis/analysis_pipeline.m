close all; clear all; clc;

%% Load settings
settings_exper;
settings_figure;
settings_analysis;
settings_screen;
settings_log;

addpath(exper.path.ANALYSIS);
cd(exper.path.ROOT);

%% Extract data from files
data.ss.log = getLogFiles(exper, anal, logCol);
data.ss.gaze = getDatFiles(exper, screen, anal, data.ss.log.nCompletedTrials);
data.ss.badTrials = ...
    getBadTrials(exper, data.ss.log.nCompletedTrials, exper.path.DATA);

%% Asses data quality
quality.excludedTrials = ...
    getExcludeTrials(exper, ...
                     anal, ...
                     data.ss.log.error.fixation.online, ...
                     data.ss.gaze.error.fixation.offline, ... 
                     data.ss.gaze.error.dataLoss, ...
                     data.ss.gaze.error.eventMissing, ...
                     data.ss.badTrials);
[quality.proportionValidTrials, quality.nValidTrials] = ...
    getProportionValidTrials(exper, anal, data.ss.log.nCompletedTrials, ...
                             quality.excludedTrials);

%% Get screen coordiantes of stimuli
data.ss.stimulusCoordinates = getStimCoord(exper, anal, logCol, data.ss.log.files);

%% Get gaze shifts
data.ss.gaze.gazeShifts = ...
    getGazeShifts(exper, anal, data.ss.gaze, data.ss.log.nCompletedTrials, ...
                  quality.excludedTrials);

%% Get fixated areas of interest
data.ss.fixations = ...
    getFixatedAois(exper, screen, anal, data.ss.gaze, ...
                   data.ss.stimulusCoordinates, ...
                   data.ss.log.nCompletedTrials, ...
                   quality.excludedTrials, ...
                   fig.toggle.debug.SHOW_FIXATIONS);

%% Get time-related variables
% Planning, inspection, and response times as well as trial durations
data.ss.time = getTimes(exper, anal, data.ss.log.nCompletedTrials, ...
                        data.ss.gaze, ...
                        data.ss.fixations, ...
                        quality.excludedTrials);

%% Get chosen target
data.ss.choice = getChoices(exper, anal, logCol, data.ss.log, data.ss.gaze, ...
                            data.ss.fixations, quality.excludedTrials);

%% DEBUG: check whether results from new match old pipeline
checkPipelines(exper, logCol, data.ss.log, data.ss.gaze, ...
               data.ss.fixations, data.ss.time, data.ss.choice, ...
               data.ss.badTrials, quality.excludedTrials, "_withExclusion");

%% Export for Zenodo
% for c = 1:exper.num.condNo % Condition
% 
%     temp = vertcat(sacc.gazeShifts_zen{:, c});
%     writematrix(temp, strcat('./dat_cond', num2str(c), '.csv'))
% 
% end
% sacc = rmfield(sacc, 'gazeShifts_zen'); 

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

    thisCondition = exper.num.conds(c);
    for s = 1:exper.num.subNo % Subject

        thisSubject  = exper.num.subs(s);
        idx_excld = sort(unique(exper.excl_trials{thisSubject, c}));
        no_valid  = exper.trialNo(thisSubject, c) - numel(idx_excld); % # valid trials

        % Calculat proportion trials for which we could calculate the decision time
        time_decision = sacc.time.decision{thisSubject, c};

        exper.prop.resp_trials(thisSubject, c) = sum(~isnan(time_decision)) / no_valid;
        clear time_decision

        % Calculate proportion trials in which the last fixated and the
        % responded on target corresponded
        if mod(thisCondition, 2) == 1 % Only doube-target condition

            no_correspond = sum(stim.choiceCorrespond{thisSubject, c} == 1); % # trials with correspondence

            exper.prop.correspond_trials(thisSubject) = no_correspond / no_valid;
            clear no_valid no_correspond

        end
        clear idx_excld

    end
    clear s thisSubject

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
clear c thisCondition

% Exclude subjects based on defined criteria
% 
% Hardcode exclusion of participant 19
% This one peformed very poorly in the double-target condition of the
% eye tracking experiment (having a negative final score), while also
% having a comparetively large number of excluded trials in this
% condition this one is excluded because of an excessive search time,
% which, however, is only calculated further down in the pipeline
% 
% Hardcode exclusion of participant 20
% This one had problems during eye tracking calibration, so we only have
% data for the manual search condition
idx_excld = logical(sum(isnan(exper.trialNo), 2));
idx_excld(19:20) = true;

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

        thisSubject         = exper.num.subs(s);
        inp_chosenTarget = stim.chosenTarget{thisSubject, c};
        inp_hitMiss      = perf.hitMiss{thisSubject, c};
        inp_decisionTime = sacc.time.decision{thisSubject, c};
        inp_noDis        = [stim.no_easyDis{s, c} stim.no_hardDis{s, c}];

        if ~isempty(inp_chosenTarget)

            [~, perf.hitrates(thisSubject, c, 1:3), perf.hitrates_withDecTime(:, :, s, c)] = ...
                infSampling_propCorrect(inp_hitMiss, inp_chosenTarget, inp_decisionTime, inp_noDis, c);

        end
        clear thisSubject inp_chosenTarget inp_hitMiss inp_decisionTime inp_noDis

    end
    clear s

end


%% Proportion choices easy target
stim.propChoice.easy = NaN(9, exper.num.subNo, exper.num.condNo);
for c = 2:exper.num.condNo % Condition

    for s = 1:exper.num.subNo % Subject

        % Get data of subject
        thisSubject       = exper.num.subs(s);
        dat_sub_choice = stim.chosenTarget{thisSubject, c};
        dat_sub_ed     = stim.no_easyDis{thisSubject, c};

        % For each set-size, determine proportion choices easy target
        ind_ss = unique(dat_sub_ed(~isnan(dat_sub_ed)));
        no_ss  = numel(ind_ss);
        for ss = 1:no_ss

            no_trials_val  = sum(dat_sub_ed == ind_ss(ss));
            no_trials_easy = sum(dat_sub_choice == stim.identifier(1, 1) & ...
                                 dat_sub_ed == ind_ss(ss));

            stim.propChoice.easy(ss, thisSubject, c) = no_trials_easy / no_trials_val;
            clear no_trials_val no_trials_easy

        end
        clear thisSubject dat_sub_choice dat_sub_ed ind_ss no_ss ss

    end
    clear s

end
clear c


%% Proportion gaze shifts on easy set as a function of set-size
sacc.propGs.onEasy_noLock_indSs = NaN(9, exper.num.subNo, exper.num.condNo);
for c = 2:exper.num.condNo % Condition

    thisCondition = exper.num.conds(c);
    for s = 1:exper.num.subNo % Subject

        % Get data of subject and drop excluded trials
        thisSubject = exper.num.subs(s);
        dat_sub  = sacc.gazeShifts{thisSubject, c};
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
            sacc.propGs.onEasy_noLock_indSs(:, thisSubject, c) = ...
                cell2mat(cellfun(@(x) x(:, 4, 3), propGs_onEasy_noLock_indSs, ...
                         'UniformOutput', false));
            clear inp_mat inp_coiLab inp_ssGroups inp_lock propGs_onEasy_noLock_indSs

        end
        clear thisSubject

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
clear c thisCondition


%% Timecourse proportion gaze shifts on stimulus in trial
sacc.propGs.onChosen_trialBegin  = cell(exper.num.subNo, exper.num.condNo);
sacc.propGs.onEasy_trialBegin    = cell(exper.num.subNo, exper.num.condNo);
sacc.propGs.onSmaller_trialBegin = cell(exper.num.subNo, exper.num.condNo);
sacc.propGs.onCloser_trialBegin  = cell(exper.num.subNo, exper.num.condNo);
for c = 2:exper.num.condNo % Condition; only double-target

    for s = 1:exper.num.subNo % Subject

        % Get data of subject and drop excluded trials
        thisSubject = exper.num.subs(s);
        dat_sub  = sacc.gazeShifts{thisSubject, c};
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
            sacc.propGs.onChosen_trialBegin{thisSubject, c} = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups);

            % Timecourse of proportion gaze shifts to easy set
            inp_mat(:, 4) = li_gsOnEasySet;
            sacc.propGs.onEasy_trialBegin{thisSubject, c} = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups);

            % Timecourse of proportion gaze shifts to closer stimulus
            inp_mat(:, 4) = li_gsOnClosestStim;
            sacc.propGs.onCloser_trialBegin{thisSubject, c} = ... 
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

            sacc.propGs.onSmaller_trialBegin{thisSubject, c} = ... 
                infSampling_timecourseGsInAOI(inp_mat, inp_lock, ...
                                              inp_coiLab, inp_ssGroups);
            clear li_gsOnSmallerSet inp_coiLab inp_ssGroups inp_lock dat_sub_noMed no_gs inp_mat

        end
        clear thisSubject dat_sub 

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

        thisSubject = exper.num.subs(s);

        dat_chosenTarg_sub = stim.chosenTarget{thisSubject, c};
        dat_noDis_sub      = [stim.no_easyDis{thisSubject, c} stim.no_hardDis{thisSubject, c}];
        dat_planTime_sub   = sacc.time.planning{thisSubject, c};
        dat_inspTime_sub   = sacc.time.inspection{thisSubject, c};
        dat_decTime_sub    = sacc.time.decision{thisSubject, c};
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
                sacc.time.mean.planning(thisSubject, c, t)   = mean(temp(1, :), 2, 'omitnan');
                sacc.time.mean.inspection(thisSubject, c, t) = mean(temp(2, :), 2, 'omitnan');
                sacc.time.mean.decision(thisSubject, c, t)   = mean(temp(3, :), 2, 'omitnan');
                sacc.time.mean.non_search(thisSubject, c, t) = mean(temp(4, :), 2, 'omitnan');
                clear temp ss

            end
            clear t NOSS

        end
        clear thisSubject dat_noDis_sub dat_planTime_sub dat_inspTime_sub dat_decTime_sub dat_planTime_sub dat_chosenTarg_sub setSizes

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

        thisSubject   = exper.num.subs(s);
        searchTime = sacc.time.search{thisSubject, c};
        if ~isempty(searchTime)

            searchTime = sacc.time.search{thisSubject, c}(:, 4);
            noDis_sub  = [stim.no_easyDis{thisSubject, c} stim.no_hardDis{thisSubject, c}];
            no_ss      = unique(noDis_sub(~isnan(noDis_sub(:, 1)), 1));
            for ss = 1:numel(no_ss) % Set size

                switch c

                    case 1
                        li_trials = any(noDis_sub == no_ss(ss), 2);

                    case 2
                        li_trials = noDis_sub(:, 1) == no_ss(ss);

                end

                sacc.time.search_ss(thisSubject, ss, c) = mean(searchTime(li_trials), 'omitnan');
                clear li_trials

            end
            clear no_ss ss noDis_sub

            % Regression over mean inspection time for different set sizes
            reg_predictor = (0:8)';
            reg_criterion = sacc.time.search_ss(thisSubject, :, c)';

            [sacc.time.search_reg_coeff(thisSubject, :, c), sacc.time.search_confInt(:, :, thisSubject, c)] = ...
                regress(reg_criterion, [ones(numel(reg_predictor), 1) reg_predictor]);
            clear reg_predictor reg_criterion

        end
        clear thisSubject searchTime

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

        thisSubject = exper.num.subs(s);
        gs_sub   = sacc.gazeShifts{thisSubject, c};
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
        clear thisSubject gs_sub

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

        thisSubject = exper.num.subs(s);
        gs_sub   = sacc.gazeShifts{thisSubject, c};
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
        clear thisSubject gs_sub

    end
    clear s

end
clear c


%% Latencies of first movement in trial
sacc.latency.firstGs = NaN(exper.num.subNo, 3, exper.num.condNo);
for c = 1:exper.num.condNo % Condition
    for s = 1:exper.num.subNo % Subject
        thisSubject = exper.num.subs(s);
        subDat = sacc.gazeShifts{thisSubject,c};
        if ~isempty(subDat)
            % Unpack data
            latencies = subDat(:,11);
            saccNo = subDat(:,24);
            chosenTarget = subDat(:,23);
            nDisEasy = subDat(:,22);
            nDisDifficult = subDat(:,28);
            setSizes = unique(nDisEasy);
            setSizes = setSizes(~isnan(setSizes));
            nSs = numel(setSizes);

            % Find trials
            idxFirstSacc = saccNo == 1; % First saccades in trial
            idxChosenEasy = chosenTarget == stim.identifier(1,1); % Easy chosen
            idxChosenDifficult = chosenTarget == stim.identifier(1,2); % Difficult chosen

            temp = NaN(3, nSs);
            for ss = 1:numel(setSizes) % Set size
                % Single-target
                % - Select trials where either target was shown with a
                %   given number of distractor and where the easy or
                %   difficult target was shown with a given number of
                %   same-colored distractors
                % Double-target
                % - Use the number of easy distractors in a trial as
                %   reference, and find trials where a given number of easy
                %   distractors was shown and participants chose either,
                %   the easy, or the difficult target
                % - We take the number of easy distractors as reference,
                %   because nEasy == 1-nEasy or nDifficult = fliplr(nEasy)
                if c == 1 % Single-target
                    idxAnySet = any([nDisEasy, nDisDifficult] == setSizes(ss), 2);
                    idxEasySet = nDisEasy == setSizes(ss);
                    idxDifficultSet = nDisDifficult == setSizes(ss);
                elseif c == 2 % Double-target
                    idxAnySet = nDisEasy == setSizes(ss);
                    idxEasySet = idxAnySet;
                    idxDifficultSet = idxAnySet;
                end
                idxBoth = idxFirstSacc & idxAnySet;
                idxEasy = idxFirstSacc & idxEasySet & idxChosenEasy;
                idxDifficult = idxFirstSacc & idxDifficultSet & idxChosenDifficult;

                temp(:,ss) = ...
                    [median(latencies(idxBoth), 'omitnan'), ...
                     median(latencies(idxEasy), 'omitnan'), ...
                     median(latencies(idxDifficult), 'omitnan')];
                clear idxAnySet idxEasySet idxDifficultSet idxBoth idxEasy idxDifficult
            end
            clear latencies saccNo chosenTarget nDisEasy nDisDifficult
            clear setSizes nSs idxFirstSacc idxChosenEasy idxChosenDifficult ss

            sacc.latency.firstGs(s,:,c) = mean(temp, 2, 'omitnan');
            clear temp
        end
        clear thisSubject subDat
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

% Generate lookup tablet
% infSampling_generateLUT([(1:9)' (9:-1:1)'], [0 2], 4, 1)

% Run model
if exper.num.conds(1) == 2
    if exper.flag.runModel.eye
        load('modelResults_eye_propChoices_fixChosen.mat');
    else
        model = infSampling_model_main(stim, sacc, model_io, perf, exper, plt);
    end
elseif exper.num.conds(1) == 4
    if exper.flag.runModel.tablet
        load('modelResults_tablet_propChoices_fixChosen.mat');
    else
        model = infSampling_model_main(stim, sacc, model_io, perf, exper, plt);
    end
end


%% Write data to drive
if exper.num.conds(1) == 2
    filename = 'dataEye';
elseif exper.num.conds(1) == 4
    filename = 'dataTablet';
end
save([exper.name.data, '/', filename], ...
     'exper', 'model', 'model_io', 'perf', 'plt', 'sacc', 'screen', 'stim');
clear filename


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