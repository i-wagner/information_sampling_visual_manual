function propSacc_aoi_groups = ...
            infSampling_timecourseGsInAOI(sacc_lock, lockOrder, tt_lab, ss_groups, minDatP)

    % Calculate proportion saccades on chosen set as a function of the
    % location of a saccade in a saccade sequence of a trial
    % Input
    % sacc_lock:    input matrix, that contains
    %               (:, 1): position of saccade, relative to lock
    %               (:, 2): intersaccade interval in ms
    %               (:, 3): label of AOI on which saccade landed
    %               (:, 4): logical index if saccade landed on set, defined
    %                       by "set_oi" input
    %               (:, 5): number easy distractors in trial
    %               (:, 6): timestamp of saccade
    %               (:, 7): trial number
    %               (:, 8): target chosen in trial
    % lockOrder:    Lock saccades either to last saccade in a trial (1) or
    %               the beginning of trial (2)
    % tt_lab:       Labels of AOIs to use to calculate proportion saccades
    % ss_groups:    Calculate proportion saccades for groups of set-sizes;
    %               rows are individual groups, columns are set-sizes
    %               belonging to a group
    % minDatP:      minimum number of datapoints for a lock necessary to be
    %               not excluded
    % trialNo:      Number of trials a participant did
    % Output
    % propSacc_aoi: Cell array, containing, for each cell: 
    %               (:, 1): lock of saccade (positive: locked to beginning
    %                       of trial; negative: locked to last saccade in
    %                       trial)
    %               (:, 2): number of saccades at given position relativ to
    %                       lock
    %               (:, 3): number of saccades at given position relativ to
    %                       lock that landed on chosen set
    %               (:, 4): proportion saccades at given position relativ
    %                       to lock that landed on chosen set
    %               (:, 5): proportion  saccades at given position relativ
    %                       to lock that landed on not-chosen set (:, 5)
    %               Each cell corresponds to one of the analysed set-size
    %               groups; pages within cells correspond to targets (:, :,
    %               1), distractors (:, :, 2) and set (:, :, 3)

    %% Determine order of x-axis labels
    % When we lock relativ to last saccade in trial, the plot starts with
    % negative numbers and goes up to zero (i.e., last saccade in trial);
    % when we lock relativ to trial beginning, the plot starts with zero
    % (i.e., beginning of trial) and goes up to whatever number of saccades
    % after trial start we have
    if lockOrder == 1     % Relativ to last saccade in trial

        xAxis_lockOrder = -1;

    elseif lockOrder == 2 % Relativ to beginning of trial

        xAxis_lockOrder = 1;

    end


    %% Calculate proportion saccades on chosen/not-chosen target/distractor/stimulus as a function of saccades in sequence
    cat_lock    = unique(sacc_lock(:, 1)); % Saccades relative to lock
    no_cat_lock = numel(cat_lock);         % Number of saccades relative to lock
    tt_no       = size(tt_lab, 2)+1;       % Stimulus types to analyse
    group_no    = size(ss_groups, 1);      % Number of set-size groups to analyse

    propSacc_aoi_groups = cell(group_no, 1);
    for g = 1:group_no % Set-size group

        group_current = ss_groups(g, :);
        propSacc_aoi  = NaN(no_cat_lock, 5, tt_no);
        for st = 1:tt_no % Stimulus types (target/distractor/set)

            for c = 1:no_cat_lock % Saccade number relative to lock

                if st ~= 3 % Targets/distractors analysed seperately

                    % All saccades of category
                    li_points_all = sacc_lock(:, 1) == cat_lock(c) & ismember(sacc_lock(:, 5), group_current) & ...
                                    ismember(sacc_lock(:, 3), tt_lab(st, :));

                    % Find all saccades of category that correspond to the current
                    % stimulus type "st" (target or distractor) and which landed
                    % on the stimulus category that was eventually chosen, and 
                    % all saccades of category that correspond to the current
                    % stimulus type "st" (target or distractor) and which did
                    % not land on the stimulus category that was eventually chosen
                    li_points_chosen    = sacc_lock(:, 1) == cat_lock(c) & ismember(sacc_lock(:, 5), group_current) & ... % All saccades within window that landed on target/distractor
                                          ismember(sacc_lock(:, 3), tt_lab(st, :)) & sacc_lock(:, 4) == 1;                % belonging to set of interest
                    li_points_notChosen = sacc_lock(:, 1) == cat_lock(c) & ismember(sacc_lock(:, 5), group_current) & ... % All saccades within window that landed on target/distractor
                                          ismember(sacc_lock(:, 3), tt_lab(st, :)) & sacc_lock(:, 4) == 0;                % not belonging to set of interest

                elseif st == 3

                    li_points_all = sacc_lock(:, 1) == cat_lock(c) & ismember(sacc_lock(:, 5), group_current);

                    li_points_chosen    = sacc_lock(:, 1) == cat_lock(c) & ismember(sacc_lock(:, 5), group_current) & ... All saccades on category of set of interest
                                          sacc_lock(:, 4) == 1;
                    li_points_notChosen = sacc_lock(:, 1) == cat_lock(c) & ismember(sacc_lock(:, 5), group_current) & ... All saccades not on category of set of interest
                                          sacc_lock(:, 4) == 0;

                end

                % Calculate proportion saccades on stimulus form set of interest
                no_points_all       = sum(li_points_all);                  % Number saccades of category
                no_points_chosen    = sum(li_points_chosen);               % Number saccades that landed on set of interest
                no_points_notChosen = sum(li_points_notChosen);            % Number saccades that landed on set of interest
                prop_chosen         = no_points_chosen / no_points_all;    % Proportion saccades on chosen stimulus category
                pro_notCoosen       = no_points_notChosen / no_points_all; % Proportion saccades on not chosen stimulus category

                propSacc_aoi(c, :, st) = [cat_lock(c).*xAxis_lockOrder no_points_all no_points_chosen prop_chosen pro_notCoosen];

            end
            propSacc_aoi(propSacc_aoi(:, 2, st) < minDatP, :, st) = NaN; % Entries with less than the desired number of datapoints are removed

        end

        % Store for output
        propSacc_aoi_groups{g} = propSacc_aoi;

    end

end