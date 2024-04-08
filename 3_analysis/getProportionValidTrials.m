function proportionValidTrials = getProportionValidTrials(exper, nTrials, excludedTrials)

    % Calculate proportion valid trials
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    % 
    % nTrials:
    % matrix; number of trials that participants completed in conditions
    % 
    % excludedTrials:
    % matrix; number of excluded trials for each subject and condition
    %
    % Output
    % proportionValidTrials:
    % matrix; proportion of valid (i.e., not excluded) trials for each
    % subject and condition

    %% Calculat proportion valid trials
    proportionValidTrials = NaN(exper.n.SUBJECTS, exper.n.CONDITIONS);
     for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
            thisSubject.excludedTrials = excludedTrials{thisSubject.number,c};
            thisSubject.nExcludedTrials = numel(thisSubject.excludedTrials);

            proportionValidTrials(thisSubject.number,c) = ...
                1 - (thisSubject.nExcludedTrials / thisSubject.nTrials);
        end
     end
end