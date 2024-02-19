function stimPosOrdered = sortStimLoc(locationsX, locationsY, nDisEasy, nDisDifficult, nTargets, shownTarget)

    % Sorts the the stimulus locations from the .log file so that each
    % column corresponds to a fixed stimulus type.
    %
    % Explanation:
    % The stimulus locations, extracted from the .log-file, are not
    % ordered: because of this, different cells in the location vector 
    % might correspond to different stimulus types, depending on the number 
    % of easy/difficult distractors in a trial. To make our life easier, we 
    % want to order them so that each position in the location matrix is 
    % directly linked to one type of stimulus (easy/difficult 
    % target/distractor)
    %
    % Input
    % locationsX/Y: 
    % vector; coordinates of each stimulus, presented in a trial
    %
    % nDisEasy: 
    % integer; number of easy distractors in a trial
    %
    % nDisDifficult: 
    % integer; number of difficult distractors in a trial
    %
    % nTargets: 
    % integer; number of targets in a trial
    %
    % shownTarget: 
    % integer; target shown in trial. In the double-target condition, 
    % always both targets were shown, hence, this input is meaningles there
    %
    % Output
    % shownStimuliOrdered_: 
    % matrix; the x- (:,:,1) and y-coordinates (:,:,2) for the the 
    % target(s) as well as the distractors. In the double-target condition, 
    % the first entry is the location of the easy target, then the hard
    % target, the next eight entries are easy distractors and the last eight 
    % entries are hard distractors. The structure is similar in the
    % single-target condition, with the only difference that the first 
    % entry always corresponds to the shown target, irrespective of its 
    % difficulty

    %% Sort stimulus locations
    % We will sort stimulus positions so that the array has one entry for
    % each stimulus that can be shown in a trial, i.e., 2 targets, 8 easy
    % distractors, 8 difficult distractors = 18 matrix entrie
    nTrials = size(locationsX, 1);
    nDisMax = size(locationsX, 2) - 2;
    nStimTotal = 2 + (nDisMax * 2);

    stimPosOrdered = NaN(nTrials,nStimTotal,2);
    for t = 1:nTrials % Trial
        % Remove NaNs from location vectors; NaNs represent not-assigned
        % stimulus locations, when less than nine (single-target condition)
        % stimuli were shown
        wasShown = ~isnan(locationsX(t,:));
        posShown.x = locationsX(t,wasShown);
        posShown.y = locationsY(t,wasShown);

        % In the single-target condition, # of distractors can be NaN: 
        % this is because only one distractors type is shown in each trial,
        % and the # of non-shown distractors is flagged with NaN (flagging
        % them with 0, would imply that zero distractors of the non-shown
        % type were shown, which is a seperate condition). To make life
        % easier, we recode those cases to 0 for purpose of extracting
        % stimulus locations
        thisEasyDisN = nDisEasy(t);
        thisDifficultDisN = nDisDifficult(t);
        if isnan(thisEasyDisN)
            thisEasyDisN = 0;
        elseif isnan(thisDifficultDisN)
            thisDifficultDisN = 0;
        end

        % Get location of target(s)
        % In the single-target condition, the target location is always
        % stored as the last non-NaN entry in the location vector. In the
        % double-target condition, the target location depends on the
        % target difficulty
        if nTargets(t) == 1 % Single-target condition        
            if shownTarget(t) == 1 % Easy target shown
                posTarget = cat(3, ...
                                [posShown.x(end), NaN], ...
                                [posShown.y(end), NaN]);
            elseif shownTarget(t) == 2 % Difficult target shown
                posTarget = cat(3, ...
                                [NaN, posShown.x(end)], ...
                                [NaN, posShown.y(end)]);
            end
        elseif nTargets(t) == 2 % Double-target condition
            % Difficult target is always the last non-NaN entry in the
            % location vector; easy target is always second-to-last non-NaN
            % entry in the location vector
            posTarget = cat(3, ...
                            [posShown.x(end-1), posShown.x(end)], ...
                            [posShown.y(end-1), posShown.y(end)]);
        end

        % Get locations of distractors
        % The easy distractor locations are always the first "nDisEasy" 
        % entries in the position vector
        posDisEasy = NaN(1,nDisMax,2);
        if thisEasyDisN > 0
            idx = 1:thisEasyDisN;
    
            posDisEasy(1,1:thisEasyDisN,1) = posShown.x(idx);
            posDisEasy(1,1:thisEasyDisN,2) = posShown.y(idx);
        end
    
        % The difficult distractor locations are the "nDisDifficult"
        % entries after the easy distractor locations
        posDisDifficult = NaN(1,nDisMax,2);
        if thisDifficultDisN > 0
            idx = (thisEasyDisN+1):(thisDifficultDisN+thisEasyDisN);
    
            posDisDifficult(1,1:thisDifficultDisN,1) = posShown.x(idx);
            posDisDifficult(1,1:thisDifficultDisN,2) = posShown.y(idx);
        end

        % Sort stimulus locations for output
        % The first entry is the location of the easy target, then the
        % difficult target, the next eight entries are easy distractors
        % and the last eight entries are difficult distractors
        stimPosOrdered(t,:,:) = [posTarget, posDisEasy, posDisDifficult];
    end

end