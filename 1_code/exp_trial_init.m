function [epar, el] = exp_trial_init(epar, el, tn)

    %% General settings
    % Set name of trial file and random fixation interval
    epar.eye_name   = sprintf('%s//trial%d.dat', epar.exp_path, tn);
    epar.stim_frame = (epar.fix_min + rand(1, 1) .* (epar.fix_max - epar.fix_min));

    % Set color of fixation cross
    if epar.CALIB ~= 1

        epar.fixcol = round(dklcart2rgb([0.25 0 0] .* ...
                            (round(rand(1, 3)) * 2-1)) .* 255);

    elseif epar.CALIB == 1 % Black during calibration

        epar.fixcol = [0 0 0];
        epar.CALIB  = 0;

    end
    el.foregroundcolour = epar.fixcol;
    EyelinkUpdateDefaults(el);


    %% Set which stimuli we want to show in a trial
    % No. of easy/hard distractors in a trial
    epar.dist_e = epar.trials.disBlocksRand(tn, 2); % Easy
    epar.dist_d = epar.trials.disBlocksRand(tn, 3); % Hard

    % Randomly select some gap-positions for the distractors
    epar.stim.dist_e_idx = epar.stim.dist_e(randi(size(epar.stim.dist_e, 1), epar.dist_e, 1), epar.targDiff_easy); % Easy
    epar.stim.dist_d_idx = epar.stim.dist_d(randi(size(epar.stim.dist_d, 1), epar.dist_d, 1), epar.targDiff_hard); % Hard
    if isempty(epar.stim.dist_e_idx)

        epar.stim.dist_e_idx = NaN;

    elseif isempty(epar.stim.dist_d_idx)

        epar.stim.dist_d_idx = NaN;

    end

    % In Experiment 2, we always show only one target per trial
    if epar.expNo == 2

        % Randomly select the orientation of the two targets
        if epar.diff(tn) == 1 % Easy

            epar.stim.targ_idx = epar.stim.targ_e(randperm(size(epar.stim.targ_e, 1), epar.targ), epar.targDiff_easy);

            id_targ = epar.stim.idx_easyStim;

        elseif epar.diff(tn) == 2 % Hard

            epar.stim.targ_idx = epar.stim.targ_d(randperm(size(epar.stim.targ_d, 1), epar.targ), epar.targDiff_hard);

            id_targ = epar.stim.idx_hardStim;

        end

        % Generate an array with the IDs of the to-be-displayed stimuli in
        % the current trial
        epar.stim.txt_disp = [epar.stim.dist_e_idx; ...
                              epar.stim.dist_d_idx; ...
                              epar.stim.targ_idx];

        % Generate array with IDs of mask stimuli, having the same color as
        % to-be-displayed stimuli in the current trial
        no_eD = numel(epar.stim.dist_e_idx);
        no_dD = numel(epar.stim.dist_d_idx);
        no_t  = numel(epar.stim.targ_idx);

        epar.stim.txt_disp_mask = [zeros(no_eD, 1) + epar.stim.comp(epar.stim.idx_easyStim); ...
                                   zeros(no_dD, 1) + epar.stim.comp(epar.stim.idx_hardStim); ...
                                   zeros(no_t, 1)  + epar.stim.comp(id_targ)];

    % In Experiment 3, we show both targets in each trial
    elseif epar.expNo == 3

        % Since we show both targets in each trial, we have to make
        % sure the targets are shown with different orientations; for
        % this, we select an orientation for the easy target first and
        % take the not-chosen orientation for the hard target
        orientIDX_easy = randperm(size(epar.stim.targ_e, 1), epar.targ-1);
        if orientIDX_easy == 2 | orientIDX_easy == 3

            orientIDX_difficult = ...
                epar.targHor_idx(randperm(size(epar.targHor_idx, 2), epar.targ-1));

        else

            orientIDX_difficult = ...
                epar.targVert_idx(randperm(size(epar.targVert_idx, 2), epar.targ-1));

        end
        epar.stim.targ_idx_e = epar.stim.targ_e(orientIDX_easy, epar.targDiff_easy);
        epar.stim.targ_idx_d = epar.stim.targ_d(orientIDX_difficult, epar.targDiff_hard);

        % Generate an array with the to-be-displayed stimuli
        epar.stim.txt_disp = [epar.stim.dist_e_idx; ...
                              epar.stim.dist_d_idx; ...
                              epar.stim.targ_idx_e; ...
                              epar.stim.targ_idx_d];

        % Generate array with IDs of masik stimulu, havin the same color as
        % to-be-displayed stimuli in the current trial
        no_eD = numel(epar.stim.dist_e_idx);
        no_dD = numel(epar.stim.dist_d_idx);
        no_tE = numel(epar.stim.targ_idx_e);
        no_tD = numel(epar.stim.targ_idx_d);

        epar.stim.txt_disp_mask = [zeros(no_eD, 1) + epar.stim.comp(epar.stim.idx_easyStim); ...
                                   zeros(no_dD, 1) + epar.stim.comp(epar.stim.idx_hardStim); ...
                                   zeros(no_tE, 1) + epar.stim.comp(epar.stim.idx_easyStim); ...
                                   zeros(no_tD, 1) + epar.stim.comp(epar.stim.idx_hardStim)];

    end
    epar.stim.txt_disp_mask(isnan(epar.stim.txt_disp)) = NaN;


    %% Set rect for stimuli
    % # of easy/hard distractors
    epar.trials.dist_e(tn)   = epar.trials.disBlocksRand(tn, 2);        % Easy
    epar.trials.dist_d(tn)   = epar.trials.disBlocksRand(tn, 3);        % Hard
    epar.trials.dist_num(tn) = epar.trials.disBlocksRand(tn, 2) + ... % Overall
                               epar.trials.disBlocksRand(tn, 3);

    % Determine the center of the stimulus area
    stim_area_xCenter = epar.fixLoc(1);
    stim_area_yCenter = min(epar.y) + ((max(epar.y) - min(epar.y)) / 2);

    % Randomly pick x grid locations for the to-be-shown stimuli
    x_pick  = epar.fixLoc(1);
    y_pick  = epar.fixLoc(2);
    counter = 2;
    while 1

        % Randomly draw one x-/y-position
        thisX = epar.x(randi(numel(epar.x), 1));
        thisY = epar.y(randi(numel(epar.y), 1));

        x_pick(counter) = thisX;
        y_pick(counter) = thisY;

        % Calculate the distance between each point/stimulus; if the max.
        % and min distance fulfills certain criteria, pick it, otherwise
        % keep drawing
        distancesVec = [x_pick' y_pick'];
        distances    = pdistq(distancesVec);
        minDistance  = min(distances);
        if minDistance >= epar.distMin

            counter = counter + 1;

        else

            x_pick(counter) = [];
            y_pick(counter) = [];

        end

        % If we have an odd number of stimuli to-be-shown in the
        % current trial, we remove one position from the drawn
        % distribution, check if the remaining positions are
        % distributed evenly across the screen, and add the removed
        % position back into the distribution
        if mod(epar.targ + epar.trials.dist_num(tn), 2) ~= 0 && ...
           numel(x_pick(2:end)) == epar.targ + epar.trials.dist_num(tn)

            % Put the last position into a new array, and delete it
            % from the main array
            x_pick4     = x_pick(end);
            y_pick4     = y_pick(end);
            x_pick(end) = [];
            y_pick(end) = [];

            % Compare if there is an equal number of stimuli
            % above, below, left and right of the stimulus area
            if numel(x_pick(x_pick > stim_area_xCenter)) == numel(x_pick(x_pick < stim_area_xCenter)) && ...
               numel(y_pick(y_pick > stim_area_yCenter)) == numel(y_pick(y_pick < stim_area_yCenter))

                % If so, put the deleted position in its place again 
                x_pick(end+1) = x_pick4;
                y_pick(end+1) = y_pick4;

                % And get rid of the zero coordiantes
                x_pick = x_pick(2:end);
                y_pick = y_pick(2:end);
                break

            % Break the loop if there are enough positions, and those 
            % are evenly distributed across the screen; otherwise,
            % clean the drawn positions and keep drawing until the 
            % criteria are met 
            else

                counter = 2;
                x_pick  = epar.fixLoc(1);
                y_pick  = epar.fixLoc(2);

            end

        % If we are showing an even number of stimuli, we just check if
        % the drawn positions are distributed evenly across the screen
        elseif mod(epar.targ + epar.trials.dist_num(tn), 2) == 0 && ...
               numel(x_pick(2:end)) == epar.targ + epar.trials.dist_num(tn)

            if numel(x_pick(x_pick > stim_area_xCenter)) == numel(x_pick(x_pick < stim_area_xCenter)) && ...
               numel(y_pick(y_pick > stim_area_yCenter)) == numel(y_pick(y_pick < stim_area_yCenter))

                x_pick = x_pick(2:end);
                y_pick = y_pick(2:end);
                break

            else % Re-draw locations if criterion is not fulfilled

                counter = 2;
                x_pick  = epar.fixLoc(1);
                y_pick  = epar.fixLoc(2);

            end

        end

    end

    % Put the drawn stimuli positions in a new array; set the columns of
    % the stimuli, which will not be displayed, NaN
    epar.x_pick(tn, :) = [x_pick NaN(1, 10-(numel(x_pick)))];
    epar.y_pick(tn, :) = [y_pick NaN(1, 10-(numel(y_pick)))];

    % Convert drawn positions to pixel
    x = epar.fixLoc_px(1) + (x_pick ./ epar.XPIX2DEG);
    y = epar.fixLoc_px(2) - (y_pick ./ epar.YPIX2DEG);

    % Create rect for targets/distractors
    epar.tex_rect = [];
    for s = 1:epar.trials.dist_num(tn)+epar.targ % Stimulus

        epar.tex_rect(:, s) = CenterRectOnPoint([0 0 epar.pic_size epar.pic_size], ...
                                                x(s), y(s));

    end


    %% Determine on which side of the target the gap is placed
    % 1 = Bottom 
    % 2 = Top 
    % 3 = Left 
    % 4 = Right 
    if epar.expNo == 2

        epar.stim.gap(tn, 1) = determineGapLocation(epar.stim.txt_disp(end), epar);
        epar.stim.gap(tn, 2) = NaN;

    elseif epar.expNo == 3

        epar.stim.gap(tn, 1) = determineGapLocation(epar.stim.txt_disp(end-1), epar); % Easy
        epar.stim.gap(tn, 2) = determineGapLocation(epar.stim.txt_disp(end), epar);   % Hard
        
    end

end