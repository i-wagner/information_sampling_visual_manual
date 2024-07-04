function fixationOnChosenSet = onChosenSet(anal, exper, nTrials, trialMap, excludedTrials, fixatedAoisGroup, choice)

    % Checks whether fixations landed on elements from the set, chosen in a
    % given trial
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
    % nTrials:
    % matrix; number of trials that participants completed in conditions
    %
    % trialMap:
    % matrix; trial number for each entry in the corresponding gaze shift
    % matrix
    %
    % excludedTrials:
    % matrix; numbers of trials that where excluded from analysis
    % 
    % fixatedAoisGroup:
    % matrix; group IDs of fixated AOIs, for each gaze shift
    % 
    % choice:
    % matrix; chosen target in trial
    %
    % Output
    % fixationOnChosenSet:
    % matrix; Boolean whether a gaze shift target an element from the
    % eventually chosen set or not

    %% Check whether fixation target element from chosen set
    % Create matrix with IDs for different AOIs. The matrix is designed
    % such that the ID of the chosen target (1 or 2) maps directly to the
    % IDs of set elements, as stored in the ID matrix
    idMatrix = [[exper.stimulus.id.target.EASY, ...
                 exper.stimulus.id.distractor.EASY]; ...
                [exper.stimulus.id.target.DIFFICULT, ...
                  exper.stimulus.id.distractor.DIFFICULT]];

    fixationOnChosenSet = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
            thisSubject.trialMap = trialMap{thisSubject.number,c};
            thisSubject.excludedTrials = excludedTrials{thisSubject.number,c};
            thisSubject.fixatedAois = fixatedAoisGroup{thisSubject.number,c};
            thisSubject.choice = choice{thisSubject.number,c};
            if isnan(thisSubject.nTrials) | ...
               ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            thisSubject.chosenSet = [];
            for t = 1:thisSubject.nTrials % Trial
                if ismember(t, thisSubject.excludedTrials)
                    continue
                end
                thisTrial.idx = thisSubject.trialMap == t;
                thisSubject.chosenSet = ...
                    [thisSubject.chosenSet; ...
                     zeros(sum(thisTrial.idx), 2) + ...
                     idMatrix(thisSubject.choice(t),:)];
                clear thisTrial
            end

            % Store data
            fixationOnChosenSet{thisSubject.number,c} = ...
                any(thisSubject.fixatedAois == thisSubject.chosenSet, 2);
            clear thisSubject
        end
    end

end