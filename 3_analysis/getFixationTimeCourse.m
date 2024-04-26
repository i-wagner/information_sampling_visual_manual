function proportionOnAoiOfInterest = getFixationTimeCourse(exper, anal, timelock, timelockOfInterest, onAoiOfInterest, nDistractors)

    % Calculates proportion of gaze shifts that targeted some AOI of
    % interest, seperately for specific, timelocked gaze shifts in a trial
    %
    % NOTE:
    % "nDistractors" has to be gaze-shift-wise
    %
    % NOTE 2:
    % output is calculated by, first, calculating proportion for each
    % individual set-size, and second, by averaging over the resulting
    % vector
    % 
    % Input
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
    % onAoiOfInterest:
    % matrix; Booleans whether gaze shift target some AOI of interest or
    % not
    % 
    % nDistractors:
    % matrix; gaze-shift-wise numbers of distractors in trials
    % 
    % Output
    % proportionOnAoiOfInterest:
    % matrix; proportion of gaze shifts in timelock that target AOI of
    % interest

    %% Calcualte proportions of gaze to AOI
    nLocks = numel(timelockOfInterest);

    proportionOnAoiOfInterest = NaN(exper.n.SUBJECTS, nLocks, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.timelock = timelock{thisSubject.number,c};
            thisSubject.onAoiOfInterest = onAoiOfInterest{thisSubject.number,c};
            thisSubject.nDistractors = nDistractors{thisSubject.number,c};
            thisSubject.distractorLvl = ...
                unique(thisSubject.nDistractors(~isnan(thisSubject.nDistractors)));
            thisSubject.nDistractorLvl = numel(thisSubject.distractorLvl);
            if ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            thisSubject.test = NaN(thisSubject.nDistractorLvl, nLocks);
            for l = 1:nLocks % Timelocks
                for d = 1:thisSubject.nDistractorLvl % Number of distractors
                    thisLock.isTimelockGazeShift = ...
                        thisSubject.timelock == timelockOfInterest(l);
                    thisLock.isDistractorNumber = ...
                        thisSubject.nDistractors == thisSubject.distractorLvl(d);
                    thisLock.onAoiOfInterest = ...
                        thisLock.isTimelockGazeShift & ...
                        thisLock.isDistractorNumber & ...
                        thisSubject.onAoiOfInterest;
                    thisLock.nTrials = sum(thisLock.isTimelockGazeShift & ...
                                           thisLock.isDistractorNumber);
    
                    thisSubject.test(d,l) = ...
                        sum(thisLock.onAoiOfInterest) / thisLock.nTrials;
                end
                clear thisLock
            end

            % Store data
            proportionOnAoiOfInterest(thisSubject.number,:,c) = ...
                mean(thisSubject.test, 1, 'omitnan');
            clear thisSubject
        end
    end
end