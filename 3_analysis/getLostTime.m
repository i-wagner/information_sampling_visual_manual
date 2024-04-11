function lostTime = getLostTime(exper, anal, excludedTrials, trialDurations)

    % Calculate the time lost due to trial exclusions
    %
    % NOTE:
    % The lost time is just the sum over the duration of all excluded
    % trials
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
    % lostTime:
    % matrix; time lost due to trial exclusions

    %% Calculate proportion of valid trials with response times
    lostTime = NaN(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubjects.idxExcludedTrials = ...
                excludedTrials{thisSubject.number,c};
            if ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            % Preassign NaN to make sure that participants, for which no
            % data is available, get a NaN in the lostTime matrix; taking
            % the sum over an empty variable returns zero, which is not
            % ambigous in regard to whether a subject is excluded from
            % analysis or for which just no excluded trials exist
            durationsExcludedTrials = NaN;
            if ~isempty(trialDurations{thisSubject.number,c})
                durationsExcludedTrials = ...
                    trialDurations{thisSubject.number,c}(thisSubjects.idxExcludedTrials);
            end
            lostTime(thisSubject.number,c) = sum(durationsExcludedTrials) / 1000;
        end    
    end
end