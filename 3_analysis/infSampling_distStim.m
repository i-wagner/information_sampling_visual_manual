function [li_gsToClosest, prop_gsClosest, prop_gsFurther, dist_out] = infSampling_distStim(gsOnset_x, gsOnset_y, targ_Aoi, stimLoc_x, stimLoc_y, flagBg)

    % Calculates euclidean distance between current fixation location and
    % each stimulus on the screen and determine proportion gaze shifts that
    % went to the stimulus closest to the fixation location
    % Input
    % gsOnset_x:      vector with x-coordinates of gaze shift onsets
    % gsOnset_y:      vector with y-coordinates of gaze shift onsets
    % targAoi:        vector with index of stimulus, on which each gaze shift
    %                 landed; the index refers to the position of the stimulus
    %                 in the "stimLoc_*" matrix
    % stimLoc_x:      matrix with x-location of each stimulus; columns are
    %                 individual stimuli
    % stimLoc_y:      matrix with y-location of each stimulus; columns are
    %                 individual stimuli
    % flagBg:         flag, which marks the screen background as gaze shift
    %                 target
    % Output
    % li_gsToClosest: vector with logical indices, flagging for each gaze
    %                 shift, provided as input, went to closest AOI ("1")
    %                 or not ("0")
    % prop_gs_:       proportion of gaze shifts that went to closest/more
    %                 distant stimulus
    % dist_out:       matrix with:
    %                 (:, 1:2): coordinates of gaue shift
    %                 (:, 3):   index of stimulus, closesest to gaze shift
    %                           coordinates
    %                 (:, 4):   index of gaze shift target
    %                 (:, 5:6): gaze shift went to closest/further away
    %                           stimulus
    %                 (:, 7):  distance to closest stimulus

    %% Create vector with currently fixated stimulus
    % We assume that the fixated AOI only changes due to a gaze shift.
    % Consequently, we can, for each gaze shift, determine a currently
    % fixated AOI: this just corresponds to the target of the previous gaze
    % shift. The first entry in the vector with currently fixated stimui
    % has to be NaN, since the first gaze shift was executed from the screen
    % center, where only the fixation cross is shown. For the last gaze
    % shift, we ommit the currently fixated stimulus, since there is no
    % gaze shift after the last one and it does not make much sense
    % checking if the gaze shift after the last one went to the closest
    % stimulus ...
    currFix_Aoi = [NaN; targ_Aoi(1:end-1)];


    %% Determine if gaze shif went to stimulus, closest to point of gaze shift onset
    % For this, determine for each detected gaze shift the distance to each
    % stimulus on the screen. Then, check to which stimulus the gaze shift
    % went and if this stimulus corresponds to the one that was closest to
    % the point of onset
    no_stim  = numel(stimLoc_x); % Number of stimuli
    no_gs    = numel(gsOnset_x); % Number gaze shifts
    dist_out = NaN(no_gs, 7);    % Output
    for gs = 1:no_gs % Gaze shift

        % Determine distance between onset point of current gaze shift "s"
        % and each stimulus shown on the screen
        dists_trial = NaN(no_stim, 1);
        for s_loc = 1:no_stim % Position of stimulus in location matrix

            dist_mat = [gsOnset_x(gs)    gsOnset_y(gs); ...
                        stimLoc_x(s_loc) stimLoc_y(s_loc)];

            dists_trial(s_loc) = pdist(dist_mat, 'euclidean');

        end

        % We do not care about the distance to the stimulus, within whose
        % AOI the eye currently resides. For the first saccade in a trial,
        % the eye is at screen center and not on any AOI, so we do not have
        % to exclude the currently fixated AOI for this gs
        if ~isnan(currFix_Aoi(gs)) && currFix_Aoi(gs) ~= flagBg

            dists_trial(currFix_Aoi(gs)) = NaN;

        end

        % Find index of closest stimulus
        [~, idx_closestStim] = min(dists_trial);
        if isempty(idx_closestStim)

            idx_closestStim  = NaN; % Only one stimulus shown in trial
            dist_closestStim = NaN;

        else

            dist_closestStim = dists_trial(idx_closestStim);

        end

        % Collect data for output
        dist_out(gs, 1) = gsOnset_x(gs);                            % x-coordinate onset
        dist_out(gs, 2) = gsOnset_y(gs);                            % y-coordinate onset
        dist_out(gs, 3) = idx_closestStim;                          % Index of stimulus closest to point of onset
        dist_out(gs, 4) = targ_Aoi(gs);                             % Index of gaze shift target
        dist_out(gs, 5) = dist_out(gs, 3) == dist_out(gs, 4);       % Gaze shift went to closest stimulus/did not went to it
        dist_out(gs, 6) = dist_out(gs, 3) ~= dist_out(gs, 4) && ...
                          dist_out(gs, 4) ~= flagBg;
        dist_out(gs, 7) = dist_closestStim;                         % Distance to closest stimulus
        if isnan(dist_out(gs, 3))                                   % Set indices for gaze shifts to closest/further away stimulus NaN for cases when we only have one stimulus, which is currently fixated

            dist_out(gs, 5:6) = NaN;

        end

    end

    % Set output variable NaN if no gaze shift was given as input
    if all(isnan([gsOnset_x; gsOnset_y]))

        li_gsToClosest = NaN;

    else

        li_gsToClosest = dist_out(:, 5);

    end


    %% Calculate proportion gaze shifts to closests/more distant stimulus
    no_gs        = sum(all(~isnan(dist_out), 2));
    no_gsClosest = sum(dist_out(:, 5) == 1);
    no_gsFurther = sum(dist_out(:, 6) == 1);

    prop_gsClosest = no_gsClosest / no_gs;
    prop_gsFurther = no_gsFurther / no_gs;

end