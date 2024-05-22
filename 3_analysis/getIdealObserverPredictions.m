function [gain, relativeGain, propChoicesEasy] = getIdealObserverPredictions(exper, idealObserver)

    % Generates ideal obsver predictions, based on empirical data
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % idealObserver:
    % structure; ideal observer model settings, as returned by the
    % "initIdealObserverModel" function
    %
    % Output
    % gain:
    % matrix; predicted absolute ideal-observer gain across subjects,
    % conditions, set sizes and target options
    %
    % relativeGain:
    % matrix; predicted relative ideal-observer gain across subjects,
    % conditions, and set sizes. Calcualted as gain_difficult - gain_easy,
    % i.e., lower (more negative) values correspond to higher gain for the
    % easy target
    %
    % propChoicesEasy:
    % matrix; predicted proportion choices for the easy target across
    % subjects, conditions, and set sizes

    %% Unpack input
    % Easy target
    accuracy.easy = idealObserver.input.accuracy.easy;
    inspectionTime.easy = idealObserver.input.inspectionTime.easy;
    nonSearchTime.easy = idealObserver.input.nonSearchTime.easy;
    fixations.easy = idealObserver.params.nFixations.easy;

    % Difficult target 
    accuracy.difficult = idealObserver.input.accuracy.difficult;
    inspectionTime.difficult = idealObserver.input.inspectionTime.difficult;
    nonSearchTime.difficult = idealObserver.input.nonSearchTime.difficult;
    fixations.difficult = idealObserver.params.nFixations.difficult;

    % General experiment related stuff
    payoff = idealObserver.payoff;
    nSets = numel(fixations.difficult);

    % Decision noise
    noise.sd = idealObserver.noise.sd;
    noise.nSamples = idealObserver.noise.nSamples;
    
    %% Generate model predictions
    gain = NaN(exper.n.SUBJECTS, nSets, 2, exper.n.CONDITIONS);
    relativeGain = NaN(exper.n.SUBJECTS, nSets, exper.n.CONDITIONS);
    propChoicesEasy = NaN(exper.n.SUBJECTS, nSets, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            gain(s,:,1,c) = getGain(accuracy.easy(s,c), ...
                                    inspectionTime.easy(s,c), ...
                                    nonSearchTime.easy(s,c), ...
                                    fixations.easy, ...
                                    payoff);
            gain(s,:,2,c) = getGain(accuracy.difficult(s,c), ...
                                    inspectionTime.difficult(s,c), ...
                                    nonSearchTime.difficult(s,c), ...
                                    fixations.difficult, ...
                                    payoff);
            relativeGain(s,:,c) = gain(s,:,2,c) - gain(s,:,1,c);

            propChoicesEasy(s,:,c) = ...
                predictChoices(relativeGain(s,:,c), noise.sd, noise.nSamples);
        end
    end
end