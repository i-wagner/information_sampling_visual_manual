% Analyses sequence effects
close all


%%
noGs_toAnalyse = 2;
flag_switch    = [-1 1];


%% Get proporiton trials in which chosen target was switched/not switched
prop_targetSwitching = NaN(exp.num.subNo, 2, 3);
no_trialTypes        = NaN(exp.num.subNo, 3, 3);
targetSwitching_all  = cell(exp.num.subNo, 1);
for d = 1:3 % Target difficulty

    for s = 1:exp.num.subNo % Subject

        % Calculate target switching behavior between trials
        curr_sub      = exp.num.subs(s);
        choice_target = stim.chosenTarget{curr_sub, 2}; % Chosen target
        switch_target = [NaN; diff(choice_target)];     % Target switching

        targetSwitching_all{curr_sub} = switch_target;

        % Check how often subjects switched target between trials
        % A target switch is marked by "-1" (from difficult to easy) or "1"
        % (from easy to difficult), whereas choosing the same target in the
        % current trial n as in the previous trial n-1 is marked by "0"
        if d < 3

            no_validTrials    = sum(choice_target == d & ~isnan(switch_target));
            no_constantTrials = sum(choice_target == d & switch_target == 0);
            no_switchTrials   = sum(choice_target == d & switch_target == flag_switch(d));

        else % Not seperating by target difficult

            no_validTrials    = sum(~isnan(choice_target) & ~isnan(switch_target));
            no_constantTrials = sum(switch_target == 0);
            no_switchTrials   = sum(any(switch_target == flag_switch, 2));

        end
        prop_targetSwitching(curr_sub, :, d) = [no_constantTrials/no_validTrials no_switchTrials/no_validTrials];
        if no_validTrials ~= 0

            no_trialTypes(curr_sub, :, d) = [no_constantTrials no_switchTrials no_validTrials];

        end
        clear curr_sub choice_target switch_target no_validTrials no_constantTrials no_switchTrials

    end
    clear s

    % Sanity check: proportion switch/constant trials should sum to 1
    if any(sum(prop_targetSwitching(:, :, d), 2) < 1)

        keyboard

    end

end
clear d


%% Calculate proportion gaze shifts in switch/constant trials that landed on elements from the chosen set in the current/previous trial
propGsOnChoice_thisLastTrial = NaN(2, noGs_toAnalyse, 2, exp.num.subNo);
for s = 1:exp.num.subNo % Subject

    % Get data
    curr_sub        = exp.num.subs(s);
    gazeShifts      = sacc.gazeShifts{curr_sub, 2};   % Gaze shifts
    choice_target   = stim.chosenTarget{curr_sub, 2}; % Target choice
    targetSwitching = targetSwitching_all{curr_sub};  % Switch/constant trial
    if ~isempty(gazeShifts)

        no_trials               = numel(targetSwitching);
        target_choice_thisTrial = NaN(no_trials, noGs_toAnalyse);
        target_choice_lastTrial = NaN(no_trials, noGs_toAnalyse);
        for t = 2:no_trials % Trial

            choiceInTrial = [choice_target(t) choice_target(t-1)]; % Choice in current/previous trial
            for gs = 1:noGs_toAnalyse % Gaze shift #

                % Determine if gaze shifted target element that belongs to
                % set of chosen target from the current trial t or the
                % previous trial t-1
                gsInTrial = gazeShifts(gazeShifts(:, 26) == t & ...
                                       gazeShifts(:, 24) == gs & ...
                                       gazeShifts(:, 18) ~= 666, 18);
                if ~isempty(gsInTrial) & targetSwitching(t) == 0         % Constant trial

                    target_choice_thisTrial(t, gs) = any(gsInTrial == stim.identifier(:, choiceInTrial(1)));
                    target_choice_lastTrial(t, gs) = any(gsInTrial == stim.identifier(:, choiceInTrial(2)));

                elseif ~isempty(gsInTrial) & abs(targetSwitching(t)) > 0 % Switch trial

                    target_choice_thisTrial(t, gs) = any(gsInTrial == stim.identifier(:, choiceInTrial(1)));
                    target_choice_lastTrial(t, gs) = any(gsInTrial == stim.identifier(:, choiceInTrial(2)));

                end
                clear gsInTrial

            end
            clear choiceInTrial gs

        end
        clear no_trials t

        % Calculate proportion gaze shifts in constant/switch trials that
        % targetet elements belonging to the set of the chosen target form
        % the current/previous trial
        if all(sum(isnan(target_choice_thisTrial)) ~= sum(isnan(target_choice_lastTrial))); keyboard; end
        for tt = 1:2 % Trial type

            for gs = 1:noGs_toAnalyse % Gaze shift #

                % Get number of trials from a certain type that had a
                % certain number of gaze shifts
                noTrials_withGs                   = sum(targetSwitching == 0 & ~isnan(target_choice_thisTrial(:, gs))); % All "gs" gaze shifts in contant trials
                noTrials_gsTargetChoice_thisTrial = sum(targetSwitching == 0 & target_choice_thisTrial(:, gs) == 1);    % All "gs" gaze shifts in constant trials that went to elements from the chosen set in the current trial
                noTrials_gsTargetChoice_lastTrial = sum(targetSwitching == 0 & target_choice_lastTrial(:, gs) == 1);    % All "gs" gaze shifts in constant trials that went to elements from the chosen set in the previous trial
                if tt == 2 % Switch trials

                    noTrials_withGs                   = sum(abs(targetSwitching) > 0 & ~isnan(target_choice_thisTrial(:, gs)));
                    noTrials_gsTargetChoice_thisTrial = sum(target_choice_thisTrial(:, gs) == 1 & abs(targetSwitching) == 1);
                    noTrials_gsTargetChoice_lastTrial = sum(target_choice_lastTrial(:, gs) == 1 & abs(targetSwitching) == 1);

                end

                % Calculate proportions
                propGsOnChoice_thisLastTrial(:, gs, tt, curr_sub) = [noTrials_gsTargetChoice_thisTrial / noTrials_withGs; ...
                                                                     noTrials_gsTargetChoice_lastTrial / noTrials_withGs];
                clear noTrials_withGs noTrials_gsTargetChoice_thisTrial noTrials_gsTargetChoice_lastTrial

            end
            clear gs

        end
        clear tt target_choice_thisTrial target_choice_lastTrial

    end
    clear curr_sub gazeShifts choice_target targetSwitching

end
clear s targetSwitching_all

% Sanity check: Proportions gaze shifts on switch trials should sum up to 1
% (but not in constant trials; here, we would except the same proportions 
% gaze shifts to elements of the set in the current and previous trial)
if any(squeeze(sum(propGsOnChoice_thisLastTrial(:, :, 2, :))) < 1)

    keyboard

end


%% Plots
% # constant/switch trials
subplot(1, 4, 1)
plot((1:2), no_trialTypes(:, 1:2, 3), ...
     '-o', ...
     'MarkerFaceColor', plt.color.c2, ...
     'Color',           plt.color.c2)
axis([0.50 2.50 0 max(max(no_trialTypes(:, 1:2, 3)))])
xticks(1:1:2)
xticklabels({'Constant Trials'; 'Switch trials'});
ylabel('# trials')
box off

% Proporiton trials in which chosen target was switched/not switched
subplot(1, 4, 2)
hold on
line([0.50 3.50], [0.50 0.50])
plot((1:3)+0.10, squeeze(prop_targetSwitching(:, 1, 1:3)), ...
     '-o', ...
     'MarkerFaceColor', plt.color.c2, ...
     'Color',           plt.color.c2)
plot(1:3, mean(squeeze(prop_targetSwitching(:, 1, 1:3)), 'omitnan'), ...
     '-o', ...
     'MarkerFaceColor', plt.color.black, ...
     'Color',           plt.color.black)
errorbar(1:3, mean(squeeze(prop_targetSwitching(:, 1, 1:3)), 'omitnan'), ...
         ci_mean(squeeze(prop_targetSwitching(:, 1, 1:3))), ...
         'Color', plt.color.black)
hold off
axis([0.50 3.50 0 1])
xticks(1:1:3)
yticks(0:0.25:1)
xticklabels({'Easy chosen [trial n]'; 'Difficult chosen [trial n]'; 'All data'});
ylabel('Proportion constant trials')
box off

% Proportion gaze shifts in switch/constant trials that landed on elements from the chosen set in the current/previous trial
sp_tit = {'Constant trials'; 'Switch trials'};
for sp = 3:4 % Subplot

    dat_trial_singleSub = squeeze(propGsOnChoice_thisLastTrial(1, :, sp-2, :))';
    dat_trial_mean      = mean(dat_trial_singleSub, 'omitnan');

    subplot(1, 4, sp)
    line([0.50 2.50], [0.50 0.50])
    hold on
    plot((1:2)+0.10, dat_trial_singleSub, ...
         '-o', ...
         'MarkerFaceColor', plt.color.c2, ...
         'Color',           plt.color.c2)
    plot(1:2, dat_trial_mean, ...
         '-o', ...
         'MarkerFaceColor', plt.color.black, ...
         'Color',           plt.color.black)
    errorbar(1:2, dat_trial_mean, ...
             ci_mean(dat_trial_singleSub), ...
             'Color', plt.color.black)
    hold off
    axis([0.50 2.50 0 1])
    xticks(1:1:2)
    yticks(0:0.25:1)
    xticklabels({'First gaze shift'; 'Second gaze shift'});
    ylabel('Proportion gaze shifts on chosen set [trial n]')
    title(sp_tit{sp-2})
    box off
    clear dat_trial_singleSub dat_trial_mean

end
clear sp_tit sp