function fixationOnEasySet = onEasySet(anal, exper, fixatedAoisGroup)

    % Checks whether fixations landed on elements from easy set
    %
    % NOTE:
    % there are ways to refactor this function so its less redundant with
    % other code, but for now I am too lazy and leave it as it is
    % 
    % Input
    % anal:
    % structure; various analysis settings, as returned by the
    % "settings_analysis" script
    %
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % fixatedAoisGroup:
    % matrix; group IDs of fixated AOIs, for each gaze shift
    %
    % Output
    % fixationOnEasySet:
    % matrix; Boolean whether a gaze shift targeted an element from the
    % easy set or not

    %% Check whether fixation target element from easy set
    idMatrix = [exper.stimulus.id.target.EASY, ...
                exper.stimulus.id.distractor.EASY];

    fixationOnEasySet = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.fixatedAois = fixatedAoisGroup{thisSubject.number,c};
            if all(isnan(thisSubject.fixatedAois)) | ...
               ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            % Store data
            fixationOnEasySet{thisSubject.number,c} = ...
                any(thisSubject.fixatedAois == idMatrix, 2);
            clear thisSubject
        end
    end

end