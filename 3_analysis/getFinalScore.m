function finalScores = getFinalScore(exper, anal, scores)

    % Gets final score of participants in conditions
    %
    % NOTE:
    % The final score is defined as the score that a participant in a
    % condition had after the last trial in said condition
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
    % scores:
    % matrix; trialwise scores of participants in conditions
    %
    % Output
    % finalScores:
    % matrix; final scores of participants at the end of conditions

    %% Get final scores
    finalScores = NaN(exper.n.SUBJECTS,exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            if ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end
            thisSubject.scores = scores{thisSubject.number,c};
            if ~isempty(thisSubject.scores)
                finalScores(thisSubject.number,c) = thisSubject.scores(end);
            end
            clear thisSubject
        end
    end
end