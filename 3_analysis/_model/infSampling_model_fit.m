function [out_propChoicesEasy, out_propFixChosenSet_ss, out_noFixChosenSet_ss, out_freeParameter, out_modelError, out_modelOpt] = ...
            infSampling_model_fit(emp_propChoicesEasy, emp_propFixChosenSet_ss, emp_propFixEasySet_ss, pred_gainPerfect, ...
                                  switch_simpleComplex, switch_biasFirstFix, switch_lossFunction, switch_precision)

    % Fit simple or complex model to empirical data and predict choice as
    % well as fixation behavior of subjects
    % FOR ALL INPUTS: rows are subjects, columns are set-sizes (where applicable)
    %
    % Input
    % emp_propChoicesEasy:      Empirical proportion choices easy target
    % emp_propFixChosenSet_ss:  Empirical set-size-wise proportion fixations on CHOSEN set
    % emp_propFixEasySet_ss:    Empirical set-size-wise proportion fixations on EASY set
    % pred_gainPerfect:         Gain for easy and difficult target, as
    %                           estimated by and equation that assumes an
    %                           influence of set-size and discrimination
    %                           difficulty as well as a perfect
    %                           distribution of fixations on the chosen set
    % switch_simpleComplex:     Determines if simple (1) or complex (2)
    %                           version of model to fit
    % switch_biasFirstFix:      Toggle if fixation bias should be applied
    %                           to first gaze shift (1) or not (0)
    % switch_lossFunction:      Determines which information is used to
    %                           compute loss function
    %                           1: proportion choices easy target
    %                           2: proportion choices easy target + proportion
    %                              fixations on chosen elements
    %                           3: proportion choices easy target + proportion
    %                              fixations on chosen elements + proportion
    %                              fixations on easy elements
    % switch_precision:        Numerical precision of lookup-table & model
    %                          predictions
    %
    % Output
    % out_propChoicesEasy:     Predicted proportion choices easy target
    % out_propFixChosenSet_ss: Predicted set-size-wise proportion fixations on CHOSEN set
    % out_noFixChosenSet_ss:   Predicted set-size-wise # fixations until any target is found
    % out_freeParameter:       Estimated parameter values
    % out_modelError:          RMSE
    % out_modelOpt:            Fitting options

    %% Determine data structure
    NOSUBS     = size(emp_propChoicesEasy, 1);         % # subjects
    NOSETSIZES = size(emp_propChoicesEasy, 2);         % # set sizes
    setSizes   = [(1:NOSETSIZES)' (NOSETSIZES:-1:1)'];


    %% Load lookup-table
    % Contains pre-caulculated model predictions for combinations of biases
    % and set sizes; used to speed-up the fitting procedure. Pre-calculated
    % predictions are stored with different number of decimals, which can
    % be controlled by the "precision" constant
    switch switch_biasFirstFix

        case 0 % Unbiased first fixation
            lut_biasSetSize = struct2array(load(['infSampling_unbiasedFirstFix_lut_', num2str(switch_precision), '.mat']));

        case 1 % Biased first fixation
            lut_biasSetSize = struct2array(load(['infSampling_biasedFirstFix_lut_', num2str(switch_precision), '.mat']));

    end


    %% Fitting options
    % The more complex model has an additional free parameter, which adds
    % noise to the relative gain estimates
    model.minSearch.options = optimset('MaxFunEvals', 10000, ...
                                       'MaxIter',     10000, ...
                                       'TolFun',      1e-12, ...
                                       'TolX',        1e-12);
    switch switch_simpleComplex

        case 1 % Simple model
            model.minSearch.init = 0.10;
            model.minSearch.min  = 0;
            model.minSearch.max  = 2;

        case 2 % Complex model
            model.minSearch.init = [0.10 0.10];
            model.minSearch.min  = [0    0   ];
            model.minSearch.max  = [2    2   ];

    end


    %% Fit free parameter(s)
    % The complex model is fit with parallel processing to speed things up
    clc % Clear Command Window, so we can track fitting progress

    modelError    = NaN(NOSUBS, 1);
    freeParameter = NaN(NOSUBS, numel(model.minSearch.init));
    switch switch_simpleComplex 

        case 1 % Simple model
            for s = 1:NOSUBS % Subject

                if all(~isnan(emp_propChoicesEasy(s, :)))

                    [freeParameter(s), modelError(s)] = fminsearchbnd(@infSampling_model_lossFunction, ...
                                                                      model.minSearch.init, ...
                                                                      model.minSearch.min, ...
                                                                      model.minSearch.max, ...
                                                                      model.minSearch.options, ...
                                                                      squeeze(pred_gainPerfect(s, :, :)), ...
                                                                      emp_propChoicesEasy(s, :)', ...
                                                                      emp_propFixChosenSet_ss(s, :)', ...
                                                                      emp_propFixEasySet_ss(s, :)', ...
                                                                      setSizes, ...
                                                                      switch_simpleComplex, ...
                                                                      switch_biasFirstFix, ...
                                                                      switch_lossFunction, ...
                                                                      lut_biasSetSize, ...
                                                                      switch_precision);

                end

            end

        case 2 % Complex model
            parfor s = 1:NOSUBS % Subject

                if all(~isnan(emp_propChoicesEasy(s, :)))

                    disp(['Now fitting subject ', num2str(s)]);
%                     profile on
%                     tic
                    [freeParameter(s, :), modelError(s)] = fminsearchbnd(@infSampling_model_lossFunction, ...
                                                                         model.minSearch.init, ...
                                                                         model.minSearch.min, ...
                                                                         model.minSearch.max, ...
                                                                         model.minSearch.options, ...
                                                                         squeeze(pred_gainPerfect(s, :, :)), ...
                                                                         emp_propChoicesEasy(s, :)', ...
                                                                         emp_propFixChosenSet_ss(s, :)', ...
                                                                         emp_propFixEasySet_ss(s, :)', ...
                                                                         setSizes, ...
                                                                         switch_simpleComplex, ...
                                                                         switch_biasFirstFix, ...
                                                                         switch_lossFunction, ...
                                                                         lut_biasSetSize, ...
                                                                         switch_precision);
%                     toc
%                     profile off
%                     profile viewer
%                     keyboard

                end

            end

    end


    %% Predict proportion choices easy target + fixations on chosen/not-chosen set
    pred_propChoicesEasy = NaN(NOSETSIZES, 2, NOSUBS);
    pred_noFix           = NaN(NOSETSIZES, 3, NOSUBS);
    pred_propFix         = NaN(NOSETSIZES, 2, NOSUBS);
    for s = 1:NOSUBS % Subject

        if all(~isnan(freeParameter(s, :)))

            switch switch_simpleComplex

                case 1 % Simple model
                    [pred_propChoicesEasy(:, :, s), pred_noFix(:, :, s), pred_propFix(:, :, s)] = ...
                        infSampling_model_predictChoiceAndFix_recursive_simple(setSizes, squeeze(pred_gainPerfect(s, :, :)), ...
                                                                               freeParameter(s), switch_biasFirstFix, ...
                                                                               lut_biasSetSize, switch_precision);

                case 2 % Complex model
                    [pred_propChoicesEasy(:, :, s), pred_noFix(:, :, s), pred_propFix(:, :, s)] = ...
                        infSampling_model_predictChoiceAndFix_recursive_complex(setSizes, squeeze(pred_gainPerfect(s, :, :)), ...
                                                                                freeParameter(s, :), switch_biasFirstFix, ...
                                                                                lut_biasSetSize, switch_precision);

            end

        end

    end


    %% Output
    % Same # fixations for chosen/not-chosen and easy/difficult, so does not matter which one we take
    out_propChoicesEasy     = squeeze(pred_propChoicesEasy(:, 1, :))';
    out_noFixChosenSet_ss   = squeeze(pred_noFix(:, 3, :))';
    out_propFixChosenSet_ss = squeeze(pred_propFix(:, 1, :))';
    out_freeParameter       = freeParameter;
    out_modelError          = modelError;
    out_modelOpt            = model.minSearch;

end