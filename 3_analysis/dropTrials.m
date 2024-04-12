function variableDropped = dropTrials(exper, anal, variable, excludedTrials)

    % Drops excluded trials (i.e., set them to NaN) from variable
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
    % variable:
    % matrix; variable for which to drop trials, across subjects and
    % conditions
    %
    % excludedTrials:
    % matrix; numbers of trials that where excluded from analysis
    % 
    % Output
    % variableDropped:
    % matrix; same as input "variable", but with dropped trials being set
    % to NaN

    %% Drop excluded trials
    variableDropped = cell(exper.n.SUBJECTS,exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            if ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end
            thisSubject.variable = variable{thisSubject.number,c};
            thisSubject.excludedTrials = excludedTrials{thisSubject.number,c};

            thisSubject.variable(thisSubject.excludedTrials) = NaN;
            variableDropped{thisSubject.number,c} = thisSubject.variable;
            clear thisSubject
        end
    end

end