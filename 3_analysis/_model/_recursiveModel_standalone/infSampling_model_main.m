function model = infSampling_model_main(stim, sacc, all, perf, plt)

    % Fits simple and complex models and returns fitting results
    %
    % Input
    % stim:  structure with stimulus related data
    % sacc:  structure with eye movement related data
    % all:   structure with fitting results from perfect/noise model
    % plt:   structure with general plot settings
    %
    % Output
    % model: fitting results

    %% Init
    SUBS     = size(stim.propChoice.easy(:, :, 2), 2); % # subjects
    MODS     = 2;                                      % # models
    SETSIZES = size(stim.propChoice.easy(:, :, 2), 1); % # set sizes 

    model.BIASFIRSTFIX     = 1; % Bias first (1) fixation or not (0)
    model.LOSSFUNCTION     = 2; % Use loss function that contians "proportionChoicesEasy + proportionFixationsOnChosen"
    model.CORRECTTARGETFIX = 1; % Account for target fixation when calculating gain
    model.PRECISION        = 4; % Numerical precision of look-up table & model predictions


    %% Fit model
    % Model 1: Simple model   --> one free parameter (fixation bias)
    % Model 2: Complex  model --> two free parameter (fixation bias + decision noise)
    model.propChoicesEasy = NaN(SUBS, SETSIZES, 2);
    model.propFixChosen   = NaN(SUBS, SETSIZES, 2);
    model.noFix           = NaN(SUBS, SETSIZES, 2);
    model.freeParameter   = cell(1, 2);
    model.error           = NaN(SUBS, 2);
    model.performance     = NaN(SUBS, 2);
    model.lossFun_noDp    = NaN(1, 2);
    model.options         = cell(1, 2);
    for m = 1:MODS % Simple/complex model

        % Predict choices and fixations
        inp_emp_propChoicesEasy     = stim.propChoice.easy(:, :, 2)';
        inp_emp_propFixChosenSet_ss = sacc.propGs.onAOI_modelComparision_chosenNot_ss(:, :, 2);
        inp_emp_propFixEasySet_ss   = sacc.propGs.onAOI_modelComparision_easyDiff_ss(:, :, 2);
        inp_pred_gainPerfect        = all.model.gain(:, :, :, 3);
        inp_switch_simpleComplex    = m;

        [model.propChoicesEasy(:, :, m), model.propFixChosen(:, :, m), model.noFix(:, :, m), ...
         model.freeParameter{m},         model.error(:, m),            model.options{m}] = ...
            infSampling_model_fit(inp_emp_propChoicesEasy, inp_emp_propFixChosenSet_ss, inp_emp_propFixEasySet_ss, ...
                                  inp_pred_gainPerfect,    inp_switch_simpleComplex,    model.BIASFIRSTFIX, ...
                                  model.LOSSFUNCTION,      model.PRECISION);
        clear inp_pred_gainPerfect

        % Get # datapoints, used for loss function
        switch model.LOSSFUNCTION

            case 1
                model.lossFun_noDp(m) = numel(inp_emp_propChoicesEasy(1, :));

            case 2
                model.lossFun_noDp(m) = numel([inp_emp_propChoicesEasy(1, :) ...
                                               inp_emp_propFixChosenSet_ss(1, :)]);

            case 3
                model.lossFun_noDp(m) = numel([inp_emp_propChoicesEasy(1, :) ...
                                               inp_emp_propFixChosenSet_ss(1, :) ...
                                               inp_emp_propFixEasySet_ss(1, :)]);

        end

        % Calculate gain
        inp_acc             = [perf.hitrates(:, 1, 2) perf.hitrates(:, 1, 3)];
        inp_propChoicesEasy = model.propChoicesEasy(:, :, m);
        inp_searchTime      = [(sacc.time.mean.inspection(:, 1, 2)/1000) (sacc.time.mean.inspection(:, 1, 3)/1000)];
        inp_nonSearchTime   = [(sacc.time.mean.non_search(:, 1, 2)/1000) (sacc.time.mean.non_search(:, 1, 3)/1000)];
        inp_noFix_overall   = model.noFix(:, :, m);

        model.performance(:, m) = ...
            infSampling_calculateGain(inp_acc, inp_propChoicesEasy, inp_searchTime, inp_nonSearchTime, inp_noFix_overall, model.CORRECTTARGETFIX);
        clear inp_acc inp_searchTime inp_nonSearchTime inp_noFix_overall

        % Plots
        inp_predNoise_propChoicesEasy = all.model.choices(:, :, 3);
        inp_emp_perf                  = all.data.double.perf;
        inp_pred_perf                 = model.performance(:, m);
        inp_predPerfect_perf          = all.model.perf_perfect(:, 3);
        inp_pred_propFixChosenSet_ss  = model.propFixChosen(:, :, m);
        inp_pred_propFixEasySet_ss    = NaN(size(model.propFixChosen(:, :, m)));

        infSampling_model_plots(inp_emp_propChoicesEasy,     inp_propChoicesEasy,       inp_predNoise_propChoicesEasy, ...
                                inp_emp_perf,                inp_pred_perf,             inp_predPerfect_perf, ...
                                inp_emp_propFixChosenSet_ss, inp_emp_propFixEasySet_ss, inp_pred_propFixChosenSet_ss, ...
                                inp_pred_propFixEasySet_ss,  inp_switch_simpleComplex,  plt)
        clear inp_emp_propChoicesEasy inp_propChoicesEasy inp_emp_propFixChosenSet_ss inp_emp_propFixEasySet_ss inp_switch_simpleComplex
        clear inp_predNoise_propChoicesEasy inp_emp_perf inp_pred_perf inp_predPerfect_perf inp_pred_propFixChosenSet_ss inp_pred_propFixEasySet_ss

    end
    clear m


    %% Model comparision
    model.weights = NaN(SUBS, MODS);
    model.aic     = NaN(SUBS, MODS);
    for s = 1:SUBS % Subject

        for m = 1:MODS % Model

            no_fp = numel(model.options{m}.init);
            [model.aic(s, m), ~] = informationCriterion(model.error(s, m), no_fp, model.lossFun_noDp(1));
            clear no_fp

        end
        model.weights(s, :) = informationWeights(model.aic(s, :));
        clear m

    end
    clear s

    inp_par_simple      = model.freeParameter{1};
    inp_par_complex     = model.freeParameter{2};
    inp_weights_simple  = model.weights(:, 1);
    inp_weights_complex = model.weights(:, 2);
    infSampling_model_comparison(inp_par_simple,     inp_par_complex, ...
                                 inp_weights_simple, inp_weights_complex, ...
                                 plt)
    clear inp_par_simple inp_par_complex inp_weights_simple inp_weights_complex


    %% Save fitting results
    switch model.LOSSFUNCTION

        case 1
            name_sfx = 'propChoices';

        case 2
            name_sfx = 'propChoices_fixChosen';

        case 3
            name_sfx = 'propChoices_fixChosen_fixEasy';

    end
    save(strcat('modelResults_', name_sfx, '.mat'), 'model');
    clear name_sfx

end