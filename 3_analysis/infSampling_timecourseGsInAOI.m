function propGs_aoi_groups = infSampling_timecourseGsInAOI(sacc_lock, lockOrder, tt_lab, ss_groups)

    % Calculate proportion gaze shifts to a given stimulus category of
    % interest over the course of a trial
    % Input
    % sacc_lock:    input matrix, that contains
    %               (:, 1): position of gaze shifts, relative to time-lock
    %               (:, 2): Unused legacy column; can be filled with NaN
    %               (:, 3): AOI label of gaze shift target
    %               (:, 4): logical indices if gaze shift targeted a stimulus
    %                       from the category of interest or not
    %               (:, 5): # easy distractors in trial
    %               (:, 6): timestamp of gaze shift offset
    %               (:, 7): trial number
    %               (:, 8): target chosen in trial
    % lockOrder:    Lock either to last gaze shift in a trial (1) or the
    %               beginning of trial (2)
    % tt_lab:       Labels of easy (:, 1) and difficult (:, 2) targets
    %               (1, :) and distractors (2, :)
    % ss_groups:    Set-size groups for which to calculate proportion gaze shifts;
    %               rows are individual groups, columns are set-sizes belonging
    %               to a group
    % Output
    % propSacc_aoi: Cell array, containing, for each cell: 
    %               (:, 1): lock of gaze shifts (positive: locked to beginning
    %                       of trial; negative: locked to last gaze shift in trial)
    %               (:, 2): number of gaze shifts at given position relativ to lock
    %               (:, 3): number of gaze shifts at given position relativ to
    %                       lock that target stimuli from category of interest
    %               (:, 4): proportion saccades at given position relativ
    %                       to lock that target stimuli from category of interest
    %               (:, 5): proportion  saccades at given position relativ
    %                       to lock that that did nit target stimuli from
    %                       category of interest
    %               Each cell corresponds to one of the analysed set-size
    %               groups; pages within cells correspond to targets (:, :,
    %               1), distractors (:, :, 2) and set (:, :, 3)

    %% Determine order of x-axis labels
    % When we lock relativ to last gaze shift in trial, the plot starts with
    % negative numbers and goes up to zero (i.e., last gaze shift in trial);
    % when we lock relativ to trial beginning, the plot starts with zero
    % (i.e., beginning of trial) and goes up to whatever number of gaze
    % shifts after trial start we have
    if lockOrder == 1     % Relativ to last saccade in trial

        xAxis_lockOrder = -1;

    elseif lockOrder == 2 % Relativ to beginning of trial

        xAxis_lockOrder = 1;

    end


    %% Calculate proportion gaze shifts on stimulus category of interest over course of trial
    cat_lock    = unique(sacc_lock(:, 1)); % Gaze shifts relative to lock
    no_cat_lock = numel(cat_lock);         % Number of gaze shift relative to lock
    tt_no       = size(tt_lab, 1)+1;       % Stimulus types to analyse
    group_no    = size(ss_groups, 1);      % Number of set-size groups to analyse

    propGs_aoi_groups = cell(group_no, 1);
    for g = 1:group_no % Set-size group

        group_current = ss_groups(g, :);
        propGs_aoi    = NaN(no_cat_lock, 5, tt_no);
        for st = 1:tt_no % Stimulus types (target/distractor/set)

            for c = 1:no_cat_lock % Gaze shift number relative to lock

                if st ~= 3 % Targets/distractors analysed seperately

                    % All gaze shifts of category
                    li_points_all = sacc_lock(:, 1) == cat_lock(c) & ismember(sacc_lock(:, 5), group_current) & ...
                                    ismember(sacc_lock(:, 3), tt_lab(st, :));

                    % Find all gaze shifts of category that correspond to the current
                    % stimulus type "st" (target or distractor) and which landed
                    % on the stimulus category of interest, and all gaze shifts
                    % of category that correspond to the current stimulus type 
                    % "st" (target or distractor) and which did not land on the
                    % stimulus category that was eventually chosen
                    li_points_interest    = sacc_lock(:, 1) == cat_lock(c) & ismember(sacc_lock(:, 5), group_current) & ...
                                            ismember(sacc_lock(:, 3), tt_lab(st, :)) & sacc_lock(:, 4) == 1;
                    li_points_notInterest = sacc_lock(:, 1) == cat_lock(c) & ismember(sacc_lock(:, 5), group_current) & ...
                                            ismember(sacc_lock(:, 3), tt_lab(st, :)) & sacc_lock(:, 4) == 0;

                elseif st == 3

                    li_points_all = sacc_lock(:, 1) == cat_lock(c) & ismember(sacc_lock(:, 5), group_current);

                    li_points_interest    = sacc_lock(:, 1) == cat_lock(c) & ismember(sacc_lock(:, 5), group_current) & ...
                                            sacc_lock(:, 4) == 1;
                    li_points_notInterest = sacc_lock(:, 1) == cat_lock(c) & ismember(sacc_lock(:, 5), group_current) & ...
                                            sacc_lock(:, 4) == 0;

                end

                % Calculate proportion gaze shifts on stimulus form set of interest
                no_points_all         = sum(li_points_all);                    % # gaze shifts of category
                no_points_interest    = sum(li_points_interest);               % # gaze shifts that landed on set of interest
                no_points_notInterest = sum(li_points_notInterest);            % # gaze shifts that landed on set of interest
                prop_interest         = no_points_interest / no_points_all;    % Proportion gaze shifts on stimulus category of interest
                pro_notInteret        = no_points_notInterest / no_points_all; % Proportion gaze shifts not on stimulus category of interest

                propGs_aoi(c, :, st) = [cat_lock(c).*xAxis_lockOrder no_points_all no_points_interest prop_interest pro_notInteret];

            end
            propGs_aoi(propGs_aoi(:, 2, st) < 1, :, st) = NaN; % Entries without are removed

        end
        propGs_aoi_groups{g} = propGs_aoi; % Store for output

    end

end