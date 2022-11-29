function [hitrates, hitrates_ss, hitrates_decisionTime] = infSampling_propCorrect(hitsMisses, targetInTrial, decisionTimes, inp_noDis, inp_cond)

    % Calculate a participants perceptual performance
    % Input
    % hitsMisses:            vector with indicators if a trial was a hit or a miss
    % targetInTrial:         vector with indicators, which target was chosen
    % decisionTimes:         vector containing trialwise decision times
    % inp_noDis:             # easy/difficult distractors in trial
    % inp_cond:              condition from which we analyse data
    % Output
    % hitrates:              column-vector with overall and target-specific hitrates
    %                        (1): overall
    %                        (2): easy target
    %                        (3): hard target
    % hitrates_ss:           same as "hitrates", but calculated by, first,
    %                        calculating proportion correct for each set size
    %                        individually, and second, averaging over set
    %                        sizes
    % hitrates_decisionTime: hitrates for trials were we could /could not
    %                        calculate decision times, seperately for
    %                        overall, easy target and difficult targets

    %% DEBUG
    % More missing trials in one than the other variable; might happen if
    % chosen target is determined by checking what was fixated last
    if sum(isnan(targetInTrial)) ~= sum(isnan(hitsMisses))

        keyboard

    end


    %% Flag trials without choices
    % Code we need for cases, in which chosen target is determined by
    % checking what was fixated last. Using this method, we sometimes
    % cannot obtain a chosen target, because subjects look at a distractor
    % before placing their response; to account for this, the corresponding
    % trials in the hit/miss variable have to excluded, otherwise accuracy
    % would yield weird results
    li_noChoice = isnan(targetInTrial);

    hitsMisses(li_noChoice) = NaN;
    clear li_noChoice


    %% Calculate overall hitrate and hitrate seperate for both target difficults
    % DO NOT SEPERATE FOR SET SIZES
    noTrials = sum(~isnan(targetInTrial));

    hitrates    = NaN(3, 1);
    hitrates(1) = sum(hitsMisses == 1) / noTrials;
    for d = 1:2 % Target difficulty

        noTrials = sum(targetInTrial == d);

        hitrates(d+1) = sum(hitsMisses == 1 & targetInTrial == d) / noTrials;
        clear noTrials

    end
    clear d


    %% Calculate overall hitrate and hitrate seperate for both target difficults
    % SEPERATE FOR SET SIZES
    setSizes = unique(inp_noDis(~isnan(inp_noDis(:, 1)), 1));
    NOSS     = numel(setSizes);

    hitrates_ss = NaN(3, 1);
    for d = 1:3 % Target difficulty

        temp = NaN(1, NOSS);
        for ss = 1:NOSS % Set size

            switch d % Target difficulty

                case 1
                    switch inp_cond % Single-/double-target condtion

                        case 1 % Single-target: trials where easy/difficult target was shown with given number of distractors
                            li_trials = any(inp_noDis == setSizes(ss), 2);

                        case 2 % Double-target: trials where easy/difficult was chosen with given number of same colored distractors
                            li_trials = inp_noDis(:, 1) == setSizes(ss);

                    end

                otherwise
                    switch inp_cond % Single-/double-target condtion
                        case 1
                            li_trials = inp_noDis(:, d-1) == setSizes(ss) & targetInTrial == d-1;

                        case 2
                            li_trials = inp_noDis(:, 1) == setSizes(ss) & targetInTrial == d-1;
                    end

            end
            noTrials  = sum(li_trials);
            noHits    = sum(hitsMisses(li_trials) == 1);
            temp(ss)  = noHits / noTrials;

        end
        hitrates_ss(d) = mean(temp, 'omitnan');

    end
    clear d


    %% Calculate hitrates for trials with/without decision times
    % Decision times are only calculated when the last element, fixated in
    % a trial, is a target; thus, trials with decision time correspond to
    % trials where participants looked at target while responding, whereas
    % trials without decision times correspond to trials where participants
    % looked at a distractor while responding
    noTrials_withDecisionTime = sum(~isnan(decisionTimes));
    noTrials_noDecisionTime   = sum(isnan(decisionTimes) & ~isnan(targetInTrial));

    hitrates_decisionTime       = NaN(3, 2);
    hitrates_decisionTime(1, 1) = sum(~isnan(decisionTimes) & hitsMisses == 1) / noTrials_withDecisionTime;
    hitrates_decisionTime(1, 2) = sum(isnan(decisionTimes) & ~isnan(targetInTrial) & hitsMisses == 1) / noTrials_noDecisionTime;
    for d = 1:2 % Target difficulty

        noTrials_withDecisionTime = sum(~isnan(decisionTimes) & targetInTrial == d);
        noTrials_noDecisionTime   = sum(isnan(decisionTimes) & ~isnan(targetInTrial) & targetInTrial == d);

        hitrates_decisionTime(d+1, 1) = ...
            sum(~isnan(decisionTimes) & targetInTrial == d & hitsMisses == 1) / noTrials_withDecisionTime;
        hitrates_decisionTime(d+1, 2) = ...
            sum(isnan(decisionTimes) & ~isnan(targetInTrial) & targetInTrial == d & hitsMisses == 1) / noTrials_noDecisionTime;

    end

end