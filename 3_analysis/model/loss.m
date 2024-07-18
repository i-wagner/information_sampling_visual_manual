function dev = loss(par, relativeGain, propChoiceEasy, propFixChosen, setSizes, nNoiseSamples, lut, precision)

    % Calculate loss between empirical data and model predictions
    %
    % NOTE 1:
    % Use the sum of squared deviations to calculate loss
    %
    % NOTE 2:
    % Uses both the proportion choices and proportion fixation on easy
    % targets to calculate loss
    %
    % Input
    % par:
    % vector; model parameter
    % 
    % relativeGain:
    % matrix; relative gain of participants
    % 
    % propChoiceEasy:
    % matrix; empirical proportion choices for easy targets
    % 
    % propFixChosen:
    % matrix; empirical proportion fixations on elements of the chosen set
    % 
    % setSizes:
    % matrix; number of easy (:,1) and difficult distractors (:,2) for
    % which to generate model predictions
    % 
    % nNoiseSamples:
    % int; number of noise samples to apply on gain estimates
    % 
    % lut:
    % matrix; look-up table with pre-calculated model predictions
    % 
    % precision:
    % int; numerical precision of model parameters and predictions
    %
    % Output
    % dev:
    % double; loss between empirical data and model predictions

    %% Get model predictions
    [predPropChoiceEasy, ~, predPropFix] = ...
        getModelPred(setSizes, relativeGain, par, nNoiseSamples, lut, precision);

    %% Calculte model error
    dev = sum(([propChoiceEasy; propFixChosen] - ...                                                       
               [predPropChoiceEasy; predPropFix]).^2, 'omitnan');

end