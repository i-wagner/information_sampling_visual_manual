function pos_all = infSampling_getStimLoc(locationsX, locationsY, noEasyDis, noHardDis, noOfTargets, inp_shownTarg)

    % Extract the on-screen coordinates of the stimuli, presented in a trial
    % Input
    % locationsX/Y:  vector containing the coordinates of each stimulus,
    %                presented in a trial
    % noEasyDis:     number of easy distractors in a trial
    % noHardDis:     number of hard distractors in a trial
    % noOfTargets:   number of targets in a trial
    % inp_shownTarg: target shown in trial. In the double-target condition,
    %                always both targets were shown, hence, this input is
    %                meaningles there
    % Output
    % pos_all:      column-vector, containing the x- (:, :, 1) and
    %               y-coordinates (:, :, 2) for the the target(s) as well
    %               as the distractors. In the double-target condition, the
    %               first entry is the location of the easy target, then the
    %               hard target, the next eight entries are easy distractors
    %               and the last eight entries are hard distractors. The
    %               structure is similar in the single-target condition,
    %               with the only difference that the first entry always
    %               corresponds to the shown target, irrespective of its
    %               difficulty

    %% Prepare input
    % Remove NaNs from location vectors; NaNs represent not-assigned
    % stimulus locations, when less than nine (single-target condition)
    % stimuli were shown
    li_notNan = ~isnan(locationsX);
    xPosAll   = locationsX(li_notNan);
    yPosAll   = locationsY(li_notNan);
    clear li_notNan


    %% Extract stimulus locations
    % Get location of target(s)
    % In the single-target condition, the target location is always stored
    % as the last non-NaN entry in the location vector. In the double-target
    % condition, the target location depends on the target difficulty
    if noOfTargets == 1     % Single-target condition        

        if inp_shownTarg == 1     % Easy target shown

            pos_t = cat(3, [xPosAll(end) NaN], [yPosAll(end) NaN]);

        elseif inp_shownTarg == 2 % Hard target shown

            pos_t = cat(3, [NaN xPosAll(end)], [NaN yPosAll(end)]);

        end

    elseif noOfTargets == 2 % Double-target condition

        % Hard target is always the last non-NaN entry in the location
        % vector; easy target is always second-to-last non-NaN entry in the
        % location vector
         pos_t = cat(3, [xPosAll(end-1) xPosAll(end)], ...
                        [yPosAll(end-1) yPosAll(end)]);

    end

    % Get locations of distractors
    pos_de = NaN(1, 8, 2);
    if noEasyDis ~= 0

        % The easy distractor locations are always the first "noEasyDis"
        % entries in the position vector
        idx_ed = 1:noEasyDis;
        no_ed  = 1:noEasyDis;

        pos_de(1, no_ed, 1) = xPosAll(idx_ed);
        pos_de(1, no_ed, 2) = yPosAll(idx_ed);
        clear idx_ed no_ed

    end

    pos_dd = NaN(1, 8, 2);
    if noHardDis ~= 0

        % The hard distractor locations are the "noHardDis(t)" entries after
        % the easy distractor locations
        idx_dd = noEasyDis+1:noHardDis+noEasyDis;
        no_dd  = 1:noHardDis;

        pos_dd(1, no_dd, 1) = xPosAll(idx_dd);
        pos_dd(1, no_dd, 2) = yPosAll(idx_dd);
        clear idx_dd no_dd

    end


    %% Sort stimulus locations for output
    % The first entry is the location of the easy target, then the hard
    % target, the next eight entries are easy distractors and the last
    % eight entries are hard distractors
    pos_all = [pos_t pos_de pos_dd];

end