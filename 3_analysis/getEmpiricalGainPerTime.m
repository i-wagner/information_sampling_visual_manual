function empiricalPerformance = getEmpiricalGainPerTime(exper, anal, lostTime, finalScores, excludedTrials)

    % Calculate empirical monetary performance, i.e., monetary gain per
    % unit of time
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
    % lostTime:
    % matrix; overall amount of time participants lost due to excluded
    % trials (in seconds)
    %
    % finalScores:
    % matrix; final scores at the end of conditions
    %
    % excludedTrials:
    % matrix; trial numbers of excluded trials
    %
    % Output
    % empiricalPerformance:
    % matrix; empirical monetary gain per second for participants and
    % conditions

    %% Get empirical gain per unit of time
    empiricalPerformance = NaN(exper.n.SUBJECTS,exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.lostTime = lostTime(thisSubject.number,c);
            thisSubject.finalScore = finalScores(thisSubject.number,c);
            thisSubject.nExcludedTrials = numel(excludedTrials{thisSubject.number,c});
            if ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            % For accumulated reward:
            % Add back the points which participants lost due to excluded
            % trials. We are doing this to get a value for the actually 
            % accumulated reward. We assume all of the excluded trials
            % yielded a reward (i.e., the target was discriminated
            % accurately)
            thisSubject.availableTime = ...
                getExpTime(exper.availableTime, thisSubject.lostTime, 'ms');
            thisSubject.accumulatedReward = ...
                (thisSubject.finalScore * 100) + ...
                (exper.payoff(1) * thisSubject.nExcludedTrials);

            empiricalPerformance(thisSubject.number,c) = ...
                thisSubject.accumulatedReward / ...
                (thisSubject.availableTime .* 60);
        end
    end
end