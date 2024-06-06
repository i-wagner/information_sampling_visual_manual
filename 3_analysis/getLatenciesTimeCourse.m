function latencies = ...
    getLatenciesTimeCourse(exper, anal, timelock, timelockOfInterest, movementLatencies, nDistractorsEasy, nDistractorsDifficult, avgSetSize, avg)

    % Calculates average latencies of gaze shifts, seperately for specific, 
    % timelocked gaze shifts in a trial
    %
    % NOTE:
    % output is calculated by, first, calculating latencies for each
    % individual set-size, and second, by averaging over the resulting
    % vector
    %
    % NOTE 2:
    % in the single-target condition, we are finding trials by checking the
    % number of easy and difficult distractors seperately. Since both
    % stimulus sets are shown in the double-target condition, we only check
    % the number of easy distractors there
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
    % timelock:
    % matrix; timelock of gaze shifts in trials, relative to trial start
    % 
    % timelockOfInterest:
    % vector/integer; gaze shift in timelock for which to calculate
    % proportions
    % 
    % movementLatencies:
    % matrix; latencies of movements in trials
    %
    % nDistractorsEasy:
    % matrix; gaze-shift-wise numbers of easy distractors in trials
    %
    % nDistractorsDifficult:
    % matrix; gaze-shift-wise numbers of difficult distractors in trials
    % 
    % Output
    % latencies:
    % matrix; timecourse of gaze shift latencies

    %% Check input
    assert(any(strcmp(avgSetSize, ["mean", "median"])));
    assert(any(strcmp(avg, ["mean", "median"])));

    %% Get latencies
    nLocks = numel(timelockOfInterest);

    latencies = NaN(exper.n.SUBJECTS, nLocks, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.timelock = timelock{thisSubject.number,c};
            thisSubject.gazeShiftLatencies = movementLatencies{thisSubject.number,c};
            thisSubject.nDistractorsEasy = nDistractorsEasy{thisSubject.number,c};
            thisSubject.nDistractorsDifficult = nDistractorsDifficult{thisSubject.number,c};
            thisSubject.distractorLvl = ...
                unique(thisSubject.nDistractorsEasy(~isnan(thisSubject.nDistractorsEasy)));
            thisSubject.nDistractorLvl = numel(thisSubject.distractorLvl);
            if ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            thisSubject.latencies = NaN(thisSubject.nDistractorLvl, nLocks);
            for l = 1:nLocks % Timelocks
                for d = 1:thisSubject.nDistractorLvl % Number of distractors
                    thisLock.isTimelockGazeShift = ...
                        thisSubject.timelock == timelockOfInterest(l);
                    thisLock.isDistractorNumber = ...
                        thisSubject.nDistractorsEasy == thisSubject.distractorLvl(d);
                    if ismember(c, [1, 3])
                        % In single-target conditions: analyse both sets
                        % together
                        thisLock.isDistractorNumber = ...
                            sum([thisSubject.nDistractorsEasy, ...
                                 thisSubject.nDistractorsDifficult], ...
                                2, 'omitnan') == ...
                            thisSubject.distractorLvl(d);
                    end
                    thisLock.gazeShifts = ...
                        all([thisLock.isTimelockGazeShift, ...
                             thisLock.isDistractorNumber], 2);
    
                    if strcmp(avgSetSize, "median")
                        thisSubject.latencies(d,l) = ...
                            median(thisSubject.gazeShiftLatencies(thisLock.gazeShifts), 'omitnan');
                    elseif strcmp(avgSetSize, "mean")
                        thisSubject.latencies(d,l) = ...
                            mean(thisSubject.gazeShiftLatencies(thisLock.gazeShifts), 'omitnan');
                    end
                    clear thisLock
                end
            end

            % Store data
            if strcmp(avg, "median")
                latencies(thisSubject.number,:,c) = ...
                    median(thisSubject.latencies, 1, 'omitnan');
            elseif strcmp(avg, "mean")
                latencies(thisSubject.number,:,c) = ...
                    mean(thisSubject.latencies, 1, 'omitnan');
            end
            clear thisSubject
        end
    end

    % If latencies are calculated for only one timelock, the second
    % dimension of the output matrix is a singleton and the output can be
    % simplified
    latencies = squeeze(latencies);
end