function [predPropChoiceEasy, predPropFix, predNFix, freeParameter] = ...
            fitProbabilisticModel(exper, anal, empChoiceEasy, empFixChosen, relativeGain, opt)

    % Fits the generative stochastic model to data of participants in the
    % double-target conditions
    %
    % NOTE 1:
    % model predictions are not actually calculated, but extracted from a
    % look-up table
    %
    % NOTE 2:
    % this function uses a parallel for loop to drastically speed up model
    % fitting
    %
    % NOTE 3:
    % unlike other functions, this function returns results for only one
    % condition at a time
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % anal:
    % structure; various analysis settings, as returned by the
    % "settings_analysis" script
    % 
    % empChoiceEasy:
    % matrix; empirical proportion choices for easy targets
    % 
    % empFixChosen:
    % matrix; empirical proportion fixations on elements of the chosen set
    % 
    % relativeGain:
    % matrix; relative gain of participants
    % 
    % opt:
    % structure; options for model fitting. Has to have the following
    % fields:
    % - BIAS_FIRST_FIX: boolean; apply fixation bias to first fixation
    % - CORRECT_TARGET_FIX: boolean; correct for target fixation
    % - PRECISION: int; numerical precision of model parameters and
    %              predictions
    % - N_NOISE_SAMPLES: int; number of noise samples to apply on gain
    %                    estimates
    % - fit: structure; options for fminsearch
    % - lut: matrix; look-up table with pre-calculated model predictions
    %
    % Output
    % predPropChoiceEasy:
    % matrix; predicted probability to choose the easy target for different
    % set sizes
    % 
    % predPropFix:
    % matrix; predicted probability to fixate elements from the chosen set
    % for different set sizes
    % 
    % predPropFix:
    % matrix; predicted number of fixations required to find the chosen
    % target
    % 
    % freeParameter:
    % matrix; fitted free parameter, i.e., decision-noise (2) and
    % fixation-noise (1)

    %% Unpack options structure
    % So we don't have to pass the entire opt structure in the parfor-loop 
    % (reduces overhead, increases processing speed)
    parInit = opt.fit.init;
    parMin = opt.fit.min;
    parMax = opt.fit.max;
    parOpt = opt.fit.options;
    nNoiseSamples = opt.N_NOISE_SAMPLES;
    lut = opt.lut;
    precision = opt.PRECISION;
    excludedSubs = anal.excludedSubjects;

    nParameter = numel(opt.fit.init);
    nSetSizes = size(empChoiceEasy, 2);
    setSizes = [(1:nSetSizes)', (nSetSizes:-1:1)'];

    %% Fit model parameter
    freeParameter = NaN(exper.n.SUBJECTS, nParameter);
    parfor s = 1:exper.n.SUBJECTS % Subject
        if ismember(s, excludedSubs) | all(isnan(relativeGain(s,:)))
            continue
        end

        freeParameter(s,:) = fminsearchbnd(@loss, ...
                                           parInit, ...
                                           parMin , ...
                                           parMax, ...
                                           parOpt, ...
                                           relativeGain(s,:)', ...
                                           empChoiceEasy(s,:)', ...
                                           empFixChosen(s,:)', ...
                                           setSizes, ...
                                           nNoiseSamples, ...
                                           lut, ...
                                           precision);
    end

    %% Get model predictions
    predPropChoiceEasy = NaN(exper.n.SUBJECTS,nSetSizes);
    predNFix = NaN(exper.n.SUBJECTS,nSetSizes);
    predPropFix = NaN(exper.n.SUBJECTS,nSetSizes);
    for s = 1:exper.n.SUBJECTS % Subject
        thisSubject.number = exper.num.SUBJECTS(s);
        if ismember(thisSubject.number, anal.excludedSubjects) | ...
           all(isnan(relativeGain(s,:)))
            continue
        end
        [thisSubject.predPropChoiceEasy, thisSubject.predNFix, ...
         thisSubject.predPropFix] = ...
            getModelPred(setSizes, ...
                         relativeGain(thisSubject.number,:)', ...
                         freeParameter(thisSubject.number,:), ...
                         nNoiseSamples, ...
                         lut, ...
                         precision);
        predPropChoiceEasy(thisSubject.number,:) = thisSubject.predPropChoiceEasy;
        predNFix(thisSubject.number,:) = thisSubject.predNFix;
        predPropFix(thisSubject.number,:) = thisSubject.predPropFix;
        clear thisSubject
    end

end