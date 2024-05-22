function propChoicesEasy = predictChoices(relativeGain, noiseSd, nNoiseSamples)

    % Predicts proportion choices for the easy target, given some gain
    % estimates and decision-noise
    %
    % NOTE 1:
    % if noise is added, the function, first, generates "nNoiseSamples"
    % noise samples, and then adds each of them to the the relative gain,
    % resulting in "nNoiseSamples" predictions. To get the final estimate,
    % the function converts the noisy gain to a binary outcome (flagging
    % higher gain target option), and takes the average over the resulting 
    % matrix
    %
    % NOTE 2:
    % for the noise, are adding Gaussian noise. We assume a mean of zero,
    % and vary the standard deviation of the noise distribution, to vary
    % the "amount/strength" of added noise
    %
    % Input
    % relativeGain:
    % vector; relative gain (difficult - easy) across different set sizes 
    %
    % noiseSd:
    % double; standard deviation of noise that is added to empirical gain
    %
    % nNoiseSamples:
    % integer; number of noise samples to add to gain predictions
    %
    % Output
    % propChoicesEasy:
    % vector; predicted proportion choices for the easy target across
    % different set sizes

    %% Predict proportion choices for easy target
    nSets = numel(relativeGain);
    propChoicesEasy = NaN(1, nSets);
    if all(~isnan(relativeGain))
        noise = repmat(randn(nNoiseSamples, 1) .* noiseSd, [1, nSets]);
        noisyGain = relativeGain + noise;

        binaryChoice = noisyGain < 0;
        propChoicesEasy = mean(binaryChoice, 1);
    end
end