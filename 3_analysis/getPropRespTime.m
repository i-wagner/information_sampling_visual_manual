function propTrialsWithRespTime = getPropRespTime(exper, anal, responseTimes, nValidTrials)

    % Calculate proportion of valid trials with response times
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
    % responseTimes:
    % matrix; response times in trials of subjects on conditions
    %
    % nValidTrials:
    % matrix; number of valid (i.e., not excluded) trials
    %
    % Output
    % propTrialsWithRespTime:
    % matrix

    %% Calculate proportion of valid trials with response times
    propTrialsWithRespTime = NaN(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.responseTimes = responseTimes{thisSubject.number,c};
            thisSubject.nValidTrials = nValidTrials(thisSubject.number,c);
            if ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end
            respTimeVailable = ~isnan(thisSubject.responseTimes);

            propTrialsWithRespTime(thisSubject.number,c) = ...
                (sum(respTimeVailable) / thisSubject.nValidTrials);
        end    
    end
end