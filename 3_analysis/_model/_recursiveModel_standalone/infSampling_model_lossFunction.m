function dev = infSampling_model_lossFunction(freeParameter, gain, emp_propChoicesEasy, emp_fixChosen, ...
                                              emp_fixEasy, setSizes, switch_simpleComplex, switch_biasFirstFix, ...
                                              switch_lossFunction, lut_biasSetSize, precision)

    % Calculates sum of squared residuales between model predictions and
    % empirical data
    % Input
    % freeParameter:        model parameter(s); one parameter for simple,
    %                       two for complex model
    % gain:                 set-size-wise gain of target; rows are set
    %                       sizes, columns are easy/difficult target
    % emp_propChoicesEasy:  set-size-wise proportion trials in which easy
    %                       target where discriminated; rows are set-sizes
    % emp_fixChosen:        set-size-wise proportion fixations on elements
    %                       of chosen set; rows are set-sizes
    % emp_fixEasy:          set-size-wise proportion fixations on easy
    %                       elements; rows are set-sizes
    % setSizes:             matrix with relative sizes of easy/difficult
    %                       set. Rows are set sizes, columns are sets
    % switch_simpleComplex: Determines which model to fit
    %                       1: simple model with 1 free parameter
    %                       2: complex model with 2 free parameter
    %                       3: model with fixed bias and variable noise
    % switch_biasFirstFix:  Toggle if we bias first gaze shift (1) or not
    %                       (0)
    % switch_lossFunction:  Determines which information is used to compute
    %                       loss function
    %                       1: proportion choices easy target
    %                       2: proportion choices easy target + proportion
    %                          fixations on chosen elements
    %                       3: proportion choices easy target + proportion
    %                          fixations on chosen elements + proportion
    %                          fixations on easy elements
    % lut_biasSetSize:      lookup table for model predictions
    % precision:            numerical precision for biases
    % Output
    % dev:                  sum of sqaured residuals between empirical data
    %                       and predictions

    %% Predict choices and fixations
    switch switch_simpleComplex

        case 1 % Simple model
            [pred_propChoicesEasy, ~, pred_propFixations] = ...
                infSampling_model_predictChoiceAndFix_recursive_simple(setSizes, gain, freeParameter, ...
                                                                       switch_biasFirstFix, lut_biasSetSize, precision);

        case 2 % Complex model

            [pred_propChoicesEasy, ~, pred_propFixations] = ...
                infSampling_model_predictChoiceAndFix_recursive_complex(setSizes, gain, freeParameter, ...
                                                                        switch_biasFirstFix, lut_biasSetSize, precision);

    end


    %% Calculte model error
    % As default, only proportion choices for easy target are used to
    % calculate loss function. They are also used when we attempt to
    % validate model, although in that we are not really fitting anything
    % anyways (since bias is fixed at 0/2)
    switch switch_lossFunction

        case 1 % Only proportion CHOICES easy target
            dev = sum((emp_propChoicesEasy - pred_propChoicesEasy(:, 1)).^2, 'omitnan');

        case 2 % Proportion CHOICES easy target & proportions FIXATIONS chosen set
            dev = sum(([emp_propChoicesEasy;        emp_fixChosen] - ...                                                       
                       [pred_propChoicesEasy(:, 1); pred_propFixations(:, 1)]).^2, 'omitnan');

        case 3 % Proportion CHOICES easy target & proportions FIXATIONS chosen set & proportions FIXATIONS easy set
            dev = sum(([emp_propChoicesEasy;        emp_fixChosen;               emp_fixEasy] - ...
                       [pred_propChoicesEasy(:, 1); pred_propFixations(:, 1); pred_propFixations(:, 1, 1)]).^2, 'omitnan');

    end

end