function [proportionValidTrials, nValidTrials] = getProportionValidTrials(exper, anal, nTrials, excludedTrials)

    % Calculate proportion valid trials
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % anal:
    % structure; vairous analysis settings, as returned by the
    % "settings_analysis" script
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
    %
    % nValidTrials:
    % matrix; numberof trials that are considered for analysis

    %% Calculat proportion valid trials
    proportionValidTrials = NaN(exper.n.SUBJECTS, exper.n.CONDITIONS);
    nValidTrials = NaN(exper.n.SUBJECTS, exper.n.CONDITIONS);
     for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            if ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            thisSubject.nTrials = nTrials(thisSubject.number,c);
            thisSubject.excludedTrials = excludedTrials{thisSubject.number,c};
            thisSubject.nExcludedTrials = numel(thisSubject.excludedTrials);

            proportionValidTrials(thisSubject.number,c) = ...
                1 - (thisSubject.nExcludedTrials / thisSubject.nTrials);
            nValidTrials(thisSubject.number,c) = ...
                thisSubject.nTrials - thisSubject.nExcludedTrials; 
        end
     end
end