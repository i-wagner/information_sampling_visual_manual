function proportionCorrect = getPerceptualPerf(exper, anal, hitOrMiss, chosenTarget, targetId, nDistractors)

    % Calculates perceptual performance for target discriminations
    %
    % NOTE:
    % Perceptual performance is calculated by, first, calculating the
    % proportion correct trials for each set-size (i.e., numbers of
    % distractors from the set of the chosen target) seperately, and
    % second, averaging over the resulting vector. We are doing this to be
    % in line with the way handle variables in the modelling module.
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
    % hitOrMiss:
    % matrix; Boolean whether trial was hit or miss
    % 
    % chosenTarget:
    % matrix; ID of chosen target in trial
    %
    % targetId:
    % integer; ID of target for which to calculate discrimination
    % performance
    % 
    % nDistractors:
    % matrix; number of distractors from the set of the target, for which
    % we are calculating discrimination performance
    %
    % Output
    % proportionCorrect:
    % matrix; proportion trials in which target was correctly
    % discriminated, across participants and conditions

    %% Calculate discrimination performance of target
    proportionCorrect = NaN(exper.n.SUBJECTS,exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            if ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end
            thisSubject.isHit = hitOrMiss{thisSubject.number,c};
            thisSubject.chosenTarget = chosenTarget{thisSubject.number,c};
            thisSubject.nDistractors = nDistractors{thisSubject.number,c};
            thisSubject.setSize.level = ...
                unique(thisSubject.nDistractors(~isnan(thisSubject.nDistractors)));
            thisSubject.setSize.n = numel(thisSubject.setSize.level);

            thisSubject.proportionCorrect = NaN(thisSubject.setSize.n, 1);
            for n = 1:thisSubject.setSize.n % Set size
                isSetSize = thisSubject.nDistractors == thisSubject.setSize.level(n);
                isTarget = thisSubject.chosenTarget == targetId;
                hits = thisSubject.isHit & isSetSize & isTarget;

                nTrials = sum(isTarget & isSetSize);
                nHits = sum(hits);
                thisSubject.proportionCorrect(n) = nHits / nTrials;
            end
            proportionCorrect(thisSubject.number,c) = ...
                mean(thisSubject.proportionCorrect, 'omitnan');
            clear thisSubject
        end
    end

end