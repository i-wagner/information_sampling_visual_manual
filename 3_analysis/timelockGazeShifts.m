function timelock = timelockGazeShifts(exper, anal, nTrials, excludedTrials, trialMap, fixatedAoisGroup, fixationsSubset)

    % Timelocks gaze shift in trials to trial start
    %
    % NOTE:
    % timelock is only applied to non-excluded gaze shifts as well as gaze
    % shifts that landed on an AOI (i.e., NOT the background). Other gaze
    % shifts will be NaN
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
    % nTrials:
    % matrix; number of trials that participants completed in conditions
    %
    % excludedTrials:
    % matrix; numbers of trials that where excluded from analysis
    %
    % trialMap:
    % matrix; trial number for each entry in the corresponding gaze shift
    % matrix
    %
    % fixatedAoisGroup:
    % matrix; group IDs of fixated AOIs, for each gaze shift
    %
    % fixationsSubset:
    % matrix; subset of non-excluded fixations
    %
    % Ouput
    % timelock:
    % matrix; timelock for gaze shifts

    %% Timelock fixations
    timelock = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
            thisSubject.excludedTrials = excludedTrials{thisSubject.number,c};
            if isnan(thisSubject.nTrials) | ...
               ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            thisSubject.timelock = [];
            for t = 1:thisSubject.nTrials % Trial
                if ismember(t, thisSubject.excludedTrials)
                    continue
                end                
                thisTrial.idx = trialMap{thisSubject.number,c} == t;
                thisTrial.subset = ...
                    logical(fixationsSubset{thisSubject.number,c}(thisTrial.idx));
                thisTrial.fixatedAois = ...
                    fixatedAoisGroup{thisSubject.number,c}(thisTrial.idx);                
                thisTrial.idxNonBgGs = ...
                    thisTrial.subset & ...
                    (thisTrial.fixatedAois ~= exper.stimulus.id.BACKGROUND);

                thisTrial.timelock = NaN(numel(thisTrial.idxNonBgGs), 1);
                thisTrial.timelock(thisTrial.idxNonBgGs) = ...
                    1:sum(thisTrial.idxNonBgGs);
                thisSubject.timelock = ...
                    [thisSubject.timelock; thisTrial.timelock];
                clear thisTrial
            end
            
            % Store data
            timelock{thisSubject.number,c} = thisSubject.timelock;
            clear thisSubject
        end
    end
end