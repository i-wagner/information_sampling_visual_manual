function proportion = getProportions(exper, anal, variable, nValidTrials, variableType)

    % Calculates proporitons of some variable of interest
    %
    % NOTE:
    % if the variable of interest is a numeric value it is, first,
    % converted to Boolean, before proportions are calculate. Booleans are
    % determined by checking for non-nan values
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
    % variable:
    % matrix; variable (for subjects and conditions) for which to 
    % calculate proportions
    %
    % nValidTrials:
    % matrix; number of valid (i.e., not excluded) trials across subjects
    % and conditions
    %
    % variableType:
    % string; "numeric" if "variable" is numberic, can be empty otherwise
    %
    % Output
    % proportion:
    % matrix; proportion of "true" cases for each cell in the input matrix
    % "variable"

    %% Calculate proportion of valid trials with response times
    proportion = NaN(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.variable = variable{thisSubject.number,c};
            thisSubject.nValidTrials = nValidTrials(thisSubject.number,c);
            if ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            if strcmp(variableType, "numeric")
                % Turn into Boolean if variable is numeric
                thisSubject.variable = ~isnan(thisSubject.variable);
            end
            proportion(thisSubject.number,c) = ...
                sum(thisSubject.variable, 'omitnan') / thisSubject.nValidTrials;
        end    
    end
end