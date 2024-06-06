function [out_propChoicesEasy, out_fixNum, out_propFix] = ...
            infSampling_model_predictChoiceAndFix_recursive_simple(setSizes_all, gain, freeParameter, switch_biasFirstFix, lut_biasSetSize, precision)

    % Predicts set-size-wise how many fixations are required to find any
    % target, the distribution of fixations easy/difficult element and
    % element from the chosen/not-chosen set and the proportion trials in
    % which easy targets should be chosen as discrimination targets
    % SIMPLE MODEL: use one free parameter
    % Input
    % setSizes_all:        matrix with relative sizes of easy/difficult
    %                      set. Rows are set sizes, columns are sets
    % gain:                set-size-wise gain of target; rows are set
    %                      sizes, columns are easy/difficult target
    % freeParameter:       SD of the function that translates gain into fixation bias
    % switch_biasFirstFix: Bias first fixation (1) or not (0)
    % lut_biasSetSize:     lookup-table with predicted proportion choices/
    %                      proportion fixations on chosen for bias/set-size
    %                      combinations
    % precision:           desired numerical precision for bias
    % Output
    % out_propChoicesEasy: set-size-wise proportion trials in which easy
    %                      targets are predicted to be chosen
    % out_fixNum:          set-size-wise # of predicted fixations on
    %                      elements from chosen/not-chosen sets
    % out_propFix:         set-size-wise proportions of predicted fixations
    %                      on elements from chosen/not-chosen sets

    %% Init
    NOSS          = size(setSizes_all, 1);
    NOBIASES      = numel(0:10^-precision:2);
    SETSIZEOFFSET = setSizes_all(:, 1) - 1;

    gain_relative = gain(:, 2) - gain(:, 1); % Difficult minus easy
    bias_all      = round(cdf('Normal', gain_relative, 0, freeParameter).*2, precision);


    %% Predict data
    % Instead of making model predictions for each combination of
    % bias/set-size, we are extracting the corresponding predictions from a
    % lookup-table. By this, we do not need to call the expensive recursive
    % algorithm during fitting. To speed things up further, the indices for
    % specific predictions in the lookup-table are calculated directly,
    % instead of, for example, using ismember() (expensive) or logical
    % indexing (less expensive than ismember(), but still costs quite a lot
    % of time)
    out_fixNum          = NaN(NOSS, 3);
    out_propChoicesEasy = NaN(NOSS, 2);
    for ss = 1:NOSS % Set size

        setSize_single = setSizes_all(ss, :);
        bias_single    = bias_all(ss);
        entry          = round(((bias_single - 0) * 10^precision) + 1) + (SETSIZEOFFSET(setSize_single(1)) * NOBIASES);
        if all(lut_biasSetSize(entry, 1:2) == [bias_single setSize_single(1)])

            sumChoice    = lut_biasSetSize(entry, 3:4);
            sumFixChoice = lut_biasSetSize(entry, 5:7);

        else

            keyboard; % Debug; since the lookup-table is complete, the function should not be called at all
            [sumChoice, sumFixChoice, ~] = decodeRecursiveProb(bias_single, setSize_single, switch_biasFirstFix);

        end
        out_fixNum(ss, :)          = sumFixChoice; % # fixation chosen/not-chosen set & # fixations overall
        out_propChoicesEasy(ss, :) = sumChoice;    % Proportion choices easy/difficult target

    end
    out_propFix = out_fixNum(:, 1:2) ./ out_fixNum(:, 3);

end