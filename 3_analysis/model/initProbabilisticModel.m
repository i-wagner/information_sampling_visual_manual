function probabilisticModel = initProbabilisticModel(exper)

    % Inits options and settings for probabilistic model
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % Output
    % probabilisticModel:
    % structure; options and settings for probabilistic model

    %% Model options
    % To speed up the fitting procedure, a pre-computed look-up table is
    % used, from which model predictions are extracted. The look-up table 
    % contains model predictions for different values of the free model 
    % parameters. The free parameter values are rounded to a certain 
    % numerical precision, and so are the free parameters during the 
    % fitting procedure. This is necessary, because otherwise, the look-up 
    % table could theoretically contain predictions for an infinite number 
    % of free parameter values (limited by the numerical precision of the 
    % machine the model runs on)
    probabilisticModel.BIAS_FIRST_FIX = true;     % Apply fixation bias to first fixation?
    probabilisticModel.CORRECT_TARGET_FIX = true; % Correct for last fixation on the chosen target
    probabilisticModel.PRECISION = 4;             % Numerical precision of look-up table & model predictions
    probabilisticModel.N_NOISE_SAMPLES = 100000; % # noise samples to use when adding fixation noise to ideal observer predictions
    
    %% Fitting options
    % The model has two free parameters: standard deviation of the added
    % fixation noise (1) and standard deviation of the added decision noise
    % (2)
    probabilisticModel.fit.options = optimset('MaxFunEvals', 100000, ...
                                              'MaxIter', 100000, ...
                                              'TolFun', 1e-12, ...
                                              'TolX', 1e-12);
    probabilisticModel.fit.min = [0, 0];
    probabilisticModel.fit.max = [2.50, 2.50];
    probabilisticModel.fit.init = ...
        mean([probabilisticModel.fit.min; probabilisticModel.fit.max]);
    
    %% Lookup table
    switch probabilisticModel.BIAS_FIRST_FIX
        case false
            lutName = strcat(exper.path.ANALYSIS, ...
                             "model/lookupTables/lut_unbiasedFirstFix_", ...
                             num2str(probabilisticModel.PRECISION), ".mat");
        case true
            lutName = strcat(exper.path.ANALYSIS, ...
                             "model/lookupTables/lut_biasedFirstFix_", ...
                             num2str(probabilisticModel.PRECISION), ".mat");
    end
    probabilisticModel.lut = struct2array(load(lutName));
end