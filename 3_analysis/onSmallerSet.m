function fixationOnSmallerSet = onSmallerSet(anal, exper, nTrials, trialMap, excludedTrials, fixatedAoisGroup, nDistractorsEasy, nDistractorsDifficult)

    % Checks whether fixations landed on elements from the set with the
    % smaller set size
    %
    % NOTE:
    % Trials where both sets had equal size are not considered here
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
    % nDistractorsEasy:
    % matrix; number of easy distractors in trials
    %
    % nDistractorsDifficult:
    % matrix; number of difficult distractors in trials
    %
    % Output
    % fixationOnSmallerSet:
    % matrix; Boolean whether a gaze shift targeted an element from the
    % smaller set or not

    %% Check whether fixation target element from the set with smaller set size
    % Create matrix with IDs for different AOIs. The matris designed such
    % that we can easily map the smaller set to the IDs of elements from
    % the corresponding set
    idMatrix = [[exper.stimulus.id.target.EASY, ...
                 exper.stimulus.id.distractor.EASY]; ...
                [exper.stimulus.id.target.DIFFICULT, ...
                  exper.stimulus.id.distractor.DIFFICULT]];


    fixationOnSmallerSet = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
            thisSubject.trialMap = trialMap{thisSubject.number,c};
            thisSubject.excludedTrials = excludedTrials{thisSubject.number,c};
            thisSubject.fixatedAois = fixatedAoisGroup{thisSubject.number,c};
            thisSubject.nDistractors.easy = nDistractorsEasy{thisSubject.number,c};
            thisSubject.nDistractors.difficult = nDistractorsDifficult{thisSubject.number,c};
            if isnan(thisSubject.nTrials) | ...
               ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            thisSubject.smallerSet = [];
            for t = 1:thisSubject.nTrials % Trial
                if ismember(t, thisSubject.excludedTrials)
                    continue
                end
                thisTrial.gazeShiftIdx = thisSubject.trialMap == t;
                thisTrial.nDistractors = [thisSubject.nDistractors.easy(t), ...
                                          thisSubject.nDistractors.difficult(t)];
                [~, thisTrial.smallerSet] = min(thisTrial.nDistractors);
                thisTrial.isEqualSetSize = ...
                    thisTrial.nDistractors(1) == thisTrial.nDistractors(2);
                if ~thisTrial.isEqualSetSize
                    thisSubject.smallerSet = ...
                        [thisSubject.smallerSet; ...
                         zeros(sum(thisTrial.gazeShiftIdx), 2) + ...
                         idMatrix(thisTrial.smallerSet,:)];
                else
                    thisSubject.smallerSet = ...
                        [thisSubject.smallerSet; ...
                         NaN(sum(thisTrial.gazeShiftIdx), 2)];
                end
                clear thisTrial
            end

            % Store data
            fixationOnSmallerSet{thisSubject.number,c} = ...
                any(thisSubject.fixatedAois == thisSubject.smallerSet, 2);
            clear thisSubject
        end
    end

end