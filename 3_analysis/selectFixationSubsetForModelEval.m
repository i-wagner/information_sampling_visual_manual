function selectedSubset = selectFixationSubsetForModelEval(anal, exper, nTrials, trialMap, excludedTrials, fixatedAoisUnique, fixationSubset)

    % Get subset of fixations, used to calculate proportion fixation on 
    % chosen set for model evaluation
    %
    % NOTE:
    % This function selects a specific subset of fixations, based on the
    % following criteria:
    % - Same general exclusion criteria, as applied in the "getFixatedAois"
    %   function
    % - Only consider search gaze shifts, i.e., exclude target fixations
    %   that are not the last fixation in a trial, immediately before a
    %   response. If a target is fixated sometime during a trial, but no
    %   other target is fixated again, this gaze shift is treated as a
    %   search gaze shift, and thus, excluded
    % - Only consider the first AOI fixation in each trial. So, not only
    %   are consecutive gaze shifts excluded (c.f., "getFixatedAois"
    %   function), but also gaze shifts to the same AOI, which are
    %   seperated by fixations to other AOIs
    % - Exclude fixations on the background
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
    % fixatedAoisUnique:
    % matrix; unique IDs of fixated AOIs, for each gaze shift
    % 
    % fixationsSubset:
    % matrix; subset of non-excluded fixations, as returned by the
    % "getFixatedAois" function
    %
    % Output
    % selectedSubset:
    % matrix; fixation subset, selected based on the defined criteria

    %% Check whether fixation target element from chosen set
    idTarget = [exper.stimulus.id.target.EASY, ...
                exper.stimulus.id.target.DIFFICULT];

    selectedSubset = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
            thisSubject.trialMap = trialMap{thisSubject.number,c};
            thisSubject.excludedTrials = excludedTrials{thisSubject.number,c};
            thisSubject.fixatedAoisUnique = fixatedAoisUnique{thisSubject.number,c};
            thisSubject.subset = fixationSubset{thisSubject.number,c};
            if isnan(thisSubject.nTrials) | ...
               ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            thisSubject.isInSubset = false(numel(thisSubject.fixatedAoisUnique), 1);
            for t = 1:thisSubject.nTrials % Trial
                thisTrial.idx = thisSubject.trialMap == t;
                thisTrial.uniqueFix = thisSubject.fixatedAoisUnique(thisTrial.idx);
                thisTrial.uniqueFixSubset = thisTrial.uniqueFix(thisSubject.subset(thisTrial.idx));
                thisTrial.nGazeShiftsInSubset = numel(thisTrial.uniqueFixSubset);
                if ismember(t, thisSubject.excludedTrials) | ...
                   isempty(thisTrial.uniqueFixSubset)
                    continue
                end

                % Get fixations on distractors
                % Only exception: if the last fixation, immediately before
                % response, is on the target, the fixation is kept in the
                % subset
                thisTrial.isLastTargetFix = false(thisTrial.nGazeShiftsInSubset, 1);
                thisTrial.idxLastTargetFixation = ...
                    find(any(thisTrial.uniqueFixSubset == idTarget, 2), ...
                         1, 'last');
                if thisTrial.idxLastTargetFixation == thisTrial.nGazeShiftsInSubset
                    thisTrial.isLastTargetFix(thisTrial.idxLastTargetFixation) = true;
                end
                thisTrial.isValidFixation = ...
                    (~any(thisTrial.uniqueFixSubset == idTarget, 2) | ... 
                     thisTrial.isLastTargetFix) & ...
                    thisTrial.uniqueFixSubset ~= exper.stimulus.id.BACKGROUND;

                % Get the first fixation for each AOI, i.e., drop repeated
                % fixations of the same AOI
                [~, thisTrial.idxUnique] = ...
                    unique(thisTrial.uniqueFixSubset, 'stable');
                idxDuplicate = setdiff(1:thisTrial.nGazeShiftsInSubset, ...
                                       thisTrial.idxUnique);
                for d = 1:numel(idxDuplicate) % Duplicate
                    % Check whether a duplicate fixation corresponds to the
                    % last fixation on a target, immediately before
                    % response. If yes: keep the fixation in the subset
                    if ~thisTrial.isLastTargetFix(idxDuplicate(d))
                        thisTrial.isValidFixation(idxDuplicate(d)) = false;
                    end
                end

                % Store data
                thisSubject.isInSubset(thisTrial.idx & thisSubject.subset) = ...
                    thisTrial.isValidFixation;
                clear thisTrial
            end

            % Store data
            selectedSubset{thisSubject.number,c} = thisSubject.isInSubset;
            clear thisSubject
        end
    end
end