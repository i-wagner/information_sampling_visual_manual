function [badTrials, nBadTrials, propBadTrials] = getBadTrials(exper, nTrials, pathToData)

    % Checks for "bad trials" in conditions of the manual search
    % experiment
    %
    % NOTE:
    % Bad trials are defined as trials, where Jan detected
    % so-called pen-dragging, i.e., participants dragging the pen over the
    % tablet, without or only rarely lifting it. Since fixation events in
    % the manual search experiment are detected by checking for instances
    % where the pen was lifted/lowered, pen dragging causes fixations to be
    % missed
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % nTrials:
    % matrix; number of trials that participants completed in conditions
    %
    % pathToData:
    % string; path to folder where data is stored
    %
    % Ouput
    % badTrials:
    % matrix; idx of bad trials
    %
    % nBadTrials:
    % matrix; number of bad trials
    %
    % propBadTrials:
    % matrix; proportion of bad trials

    %% Checker whether current trial is in Jan's BadTrial file
    badTrialfile = readtable(strcat(pathToData, "BadTrials.csv"));

    badTrials = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    nBadTrials = NaN(exper.n.SUBJECTS, exper.n.CONDITIONS);
    propBadTrials = NaN(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        % BadTrials are only available for the manual search experiment
        if ismember(c, [1, 2])
            continue;
        elseif c == 3
            conditionString = "single";
        elseif c == 4
            conditionString = "double";
        end
     
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
            if isnan(thisSubject.nTrials)
                continue;
            end

            thisSubject.badTrials = false(thisSubject.nTrials, 1);
            for t = 1:thisSubject.nTrials % Trial                
                isBad = (table2array(badTrialfile(:,"Participant")) == s) & ...
                        (table2array(badTrialfile(:,"Trial")) == t) & ...
                        strcmp(table2array(badTrialfile(:,"Task")), conditionString);
                if sum(isBad) > 1 % Sanity check: more than bad one trial?
                    keyboard
                elseif any(isBad)
                    thisSubject.badTrials(t) = true;
                end
            end
            badTrials{thisSubject.number,c} = thisSubject.badTrials;
            nBadTrials(thisSubject.number,c) = sum(thisSubject.badTrials);
            propBadTrials(thisSubject.number,c) = ...
                nBadTrials(thisSubject.number,c) / thisSubject.nTrials;
        end
    end
end
