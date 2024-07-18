function gainPerTime = getProbabilisticGain(exper, payoff, accuracy, searchTime, nonSearchTime, predPropChoicesEasy, predNFix, correctTargetFix)

    % Get gain per time for probabilistic model
    %
    % NOTE:
    % gain is calculated a bit different for this model, compared to the
    % empirical gain and ideal observer gain. 
    %
    % exper:
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % payoff:
    % vector; payoff matric for correct (1) and incorrect (2) responses
    %
    % accuracy:
    % matrix; EMPIRICAL probability to correctly discriminate easy (:,1)
    % and difficult (:,2) targets
    %
    % searchTime:
    % matrix; EMPIRICAL search time for easy (:,1) and difficult (:,2) 
    % targets
    %
    % nonSearchTime:
    % matrix; EMPIRICAL non-search time for easy (:,1) and difficult (:,2) 
    % targets
    %
    % predPropChoicesEasy:
    % matrix; PREDICTED probability to choose an easy target for different
    % set sizes
    %
    % predNFix:
    % matrix; PREDICTED number of fixations required to find a target for
    % different set sizes
    %
    % correctTargetFix:
    % Boolean; toggele whether to correct for the last target fixation
    %
    % Output
    % gainPerTime:
    % matrix; PREDICTED average monetary gain per unit of time, across all
    % set size conditions

    %% Calculate gain
    nSetSizes = size(predPropChoicesEasy, 2);
    win = payoff(1);
    loss = payoff(2);

    gainPerTime = NaN(exper.n.SUBJECTS, nSetSizes);
    for s = 1:exper.n.SUBJECTS % Subject
        propCorrectEasy = accuracy(s,1);
        propIncorrectEasy = 1 - propCorrectEasy;
        propCorrectDifficult = accuracy(s,2);
        propIncorrectDifficult = 1 - propCorrectDifficult;
        searchTimeEasy = searchTime(s,1);
        searchTimeDifficult = searchTime(s,2);
        nonSearchTimeEasy = nonSearchTime(s,1);
        nonSearchTimeDifficult = nonSearchTime(s,2);
        for ss = 1:nSetSizes % Set size
            choicesEasy = predPropChoicesEasy(s,ss);
            choicesDifficult = 1 - choicesEasy;
            nFixations = predNFix(s,ss);
            if correctTargetFix % Correct for target fixation
                nFixations = nFixations - 1;
            end

            gainEasy = (propCorrectEasy * win * choicesEasy) + ...
                       (propIncorrectEasy * loss * choicesEasy);
            gainDifficult = (propCorrectDifficult * win * choicesDifficult) + ...
                            (propIncorrectDifficult * loss * choicesDifficult);
            gain = gainEasy + gainDifficult;

            thisSearchTime = nFixations * ...
                             ((choicesEasy * searchTimeEasy) + ...
                              (choicesDifficult * searchTimeDifficult));
            thisNonSearchTime = ...
                (choicesEasy * nonSearchTimeEasy) + ...
                (choicesDifficult * nonSearchTimeDifficult);
            time = thisSearchTime + thisNonSearchTime;

            gainPerTime(s,ss) = gain / time;
        end
    end
    gainPerTime = mean(gainPerTime, 2, 'omitnan');

end