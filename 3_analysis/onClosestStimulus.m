function fixationOnCloserStim = onClosestStimulus(anal, exper, nTrials, trialMap, excludedTrials, fixatedAoisUnique, stimulusCoordinates, gazeShiftOnsets)

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

    %% Check whether fixation target the stimulus closest to the current gaze position
    fixationOnCloserStim = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
            thisSubject.trialMap = trialMap{thisSubject.number,c};
            thisSubject.excludedTrials = excludedTrials{thisSubject.number,c};
            thisSubject.fixatedAois = fixatedAoisUnique{thisSubject.number,c};
            thisSubject.stimulusCoordinates = stimulusCoordinates{thisSubject.number,c};
            thisSubject.gazeShiftOnsets = gazeShiftOnsets{thisSubject.number,c};
            if isnan(thisSubject.nTrials) | ...
               ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            thisSubject.wentToClosest = [];
            for t = 1:thisSubject.nTrials % Trial
                if ismember(t, thisSubject.excludedTrials)
                    continue
                end
                thisTrial.gazeShiftIdx = thisSubject.trialMap == t;
                thisTrial.stimulusCoordinates = ...
                    squeeze(thisSubject.stimulusCoordinates(t,:,:));
                thisTrial.gazeShifts.onsets = ...
                    thisSubject.gazeShiftOnsets(thisTrial.gazeShiftIdx,:);
                thisTrial.fixatedAois.uniqueIds = ...
                    thisSubject.fixatedAois(thisTrial.gazeShiftIdx);

                % Check if gaze shifts went to closest stimulus.
                % We are using gaze shift onset as reference here, because we
                % want to know which stimulus was closest immediately before
                % people made the gaze shift. We are currently only calculating
                % the mean gaze position between two gaze shifts, so we cannot
                % use this as the reference (since it describes the position
                % AFTER the gaze shift was already made)
                thisTrial.wentToClosest = ...
                    getDistanceToClosestStim(thisTrial.fixatedAois.uniqueIds, ...
                                             thisTrial.stimulusCoordinates(:,1), ...
                                             thisTrial.stimulusCoordinates(:,2), ...
                                             thisTrial.gazeShifts.onsets(:,2), ...
                                             thisTrial.gazeShifts.onsets(:,3), ...
                                             exper.stimulus.id.BACKGROUND, ...
                                             true);
                thisSubject.wentToClosest = [thisSubject.wentToClosest; ...
                                             thisTrial.wentToClosest];
% thisTrial.propToClosest = ...
%     mean(thisTrial.wentToClosest(thisTrial.fixationSubset), 'omitnan');
                clear thisTrial
            end

            % Store data
            fixationOnCloserStim{thisSubject.number,c} = thisSubject.wentToClosest;
            clear thisSubject
        end
    end

end