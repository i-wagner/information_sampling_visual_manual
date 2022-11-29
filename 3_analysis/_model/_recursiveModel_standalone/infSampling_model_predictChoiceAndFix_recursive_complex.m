function [out_propChoicesEasy, out_fixNum, out_propFix] = infSampling_model_predictChoiceAndFix_recursive_complex(setSizes_all, gain, freeParameter, switch_biasFirstFix, lut_biasSetSize, precision)

    % Predicts set-size-wise how many fixations are required to find any
    % target, the distribution of fixations easy/difficult element and
    % element from the chosen/not-chosen set and the proportion trials in
    % which easy targets should be chosen as discrimination targets
    % COMPLEX MODEL: use two free parameter
    % Input
    % setSizes_all:        matrix with relative sizes of easy/difficult
    %                      set. Rows are set sizes, columns are sets
    % gain:                set-size-wise gain of target; rows are set
    %                      sizes, columns are easy/difficult target
    % freeParameter:       SD of the function that translates gain into fixation bias
    % switch_biasFirstFix: Bias first fixation (1) or not (0)
    % Output
    % out_propChoicesEasy: set-size-wise proportion trials in which easy
    %                      targets are predicted to be chosen
    % out_fixNum:          set-size-wise # of predicted fixations on
    %                      elements from chosen/not-chosen sets
    % out_propFix:         set-size-wise proportions of predicted fixations
    %                      on elements from chosen/not-chosen sets

    %% Init
    NOSS          = size(setSizes_all, 1); % # set sizes
    NOBIASES      = numel(0:10^-precision:2);
    SETSIZEOFFSET = setSizes_all(:, 1) - 1;
    NOISESAMPLES  = 100000;
    NOPREDICTIONS = NOISESAMPLES*NOSS;

    gain_relative = repmat(gain(:, 2) - gain(:, 1), 1, NOISESAMPLES);
    noise         = randn(1, NOISESAMPLES) .* freeParameter(2);
    gain_relative = gain_relative + noise;
    bias_all      = round(cdf('Normal', gain_relative, 0, freeParameter(1)).*2, precision);

    % Instead of making model predictions for each combination of
    % bias/set-size, we are extracting the corresponding predictions from a
    % lookup-table. By this, we do not need to call the expensive recursive
    % algorithm during fitting. To speed things up further, the indices for
    % specific predictions in the lookup-table are calculated directly,
    % instead of, for example, using ismember() (expensive) or logical
    % indexing (less expensive than ismember(), but still costs quite a lot
    % of time)
    vector_all  = [bias_all(:) repmat(setSizes_all, NOISESAMPLES, 1)];
    idx         = round(((vector_all(:, 1) - 0) * 10^precision) + 1) + (SETSIZEOFFSET(vector_all(:, 2)) * NOBIASES);
    predictions = lut_biasSetSize(idx, :);
    if any(any((vector_all(:, 1:2) - predictions(:, 1:2)) ~= 0)); keyboard; end


    %% Predict data
    % Since all predictions for all biases where extracted from the look-up
    % table, here we only have to average over noise-samples
    out_fixNum          = NaN(NOSS, 3);
    out_propChoicesEasy = NaN(NOSS, 2);
    for ss = 1:NOSS

        li_ss = ss:9:NOPREDICTIONS; % Faster than logical indexing

        out_propChoicesEasy(ss, :) = mean(predictions(li_ss, 3:4), 1);
        out_fixNum(ss, :)          = mean(predictions(li_ss, 5:7), 1);

    end
    out_propFix = out_fixNum(:, 1:2) ./ out_fixNum(:, 3);

end