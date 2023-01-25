function dat_singleSub = infSampling_avgPropSacc(dat_input, min_sub)

    % Extractes proportion gaze shifts to stimulus of interest for single
    % subjects and sort data by set size group and location gaze shift
    % sequence
    % Input
    % dat_input: cell array with rows being single-subject data and
    %            columns different data types (for example, locked to
    %            last gaze shift in trial/trial start or proportion
    %            gaze shifts to chosen/easy set). Each cells contains
    %            another cell array, with rows being individual
    %            set-size groups, with each of those cells containing
    %            the actuall data:
    %            (:, 1): lock index
    %            (:, 2): overall number gaze shifts for this datapoint
    %            (:, 3): number gaze shifts to stimulus of interest
    %            (:, 4): proportion gaze shifts to stimulus of interest
    %            (:, 4): proportion gaze shifts not to stimulus of interest
    %            Pages correspond to saccades to target (:, :, 1), to
    %            distractor (:, :, 2) and to set (:, :, 3)
    % min_sub:   minimum number participants required to calcualte
    %            mean; datapoints with less subjects are excluded
    % Output
    % single_dt: proportion gaze shifts on AOI for individual gaze shifts of
    %            individual subjects; rows are data from subjects, columns
    %            are different gaze shifts in trials and pages are
    %            different data types (gaze shifts to chosen target, etc.)

    %% Get max. # of gaze shifts that happened in a trial and number of subjects with data
    maxNo_gs   = horzcat(dat_input{:});                                                              % Maximum # gaze shifts
    maxNo_gs   = cellfun(@max, maxNo_gs, 'UniformOutput', false);
    maxNo_gs   = vertcat(maxNo_gs{:});
    maxNo_gs   = max(maxNo_gs(:, 1, 3));
    no_valSubs = max(sum(cell2mat(cellfun(@(x) size(x, 1) > 0, dat_input, 'UniformOutput', false)), 1)); % # subjects with data
    no_ss      = max(max(cell2mat(cellfun(@(x) size(x, 1), dat_input, 'UniformOutput', false))));        % Number set-size groups


    %% For each input data type, extract data of individual subjects
    no_dt = size(dat_input, 2);

    dat_singleSub = NaN(no_valSubs, maxNo_gs, no_ss, no_dt);
    for dt = 1:no_dt % Data type

        % Unpack from current data type "dt"
        dat_type     = dat_input(:, dt);
        dat_type     = horzcat(dat_type{:});
        dat_type_pad = cellfun(@(x) x(:, :, 3), dat_type, 'UniformOutput', false); % Pad data with zeros, so each subject matrix has the same size
        nGs = cell2mat(cellfun(@(x) size(x, 1), dat_type_pad, 'UniformOutput', false));
        nGs = max(nGs(:));
        dat_type_pad = padCells(dat_type_pad, nGs);

        % Extract data of single subjects and sort it by set-size group and
        % position in sequence
        for ss = 1:no_ss % Set-size groups

            dat_type_ss = vertcat(dat_type{ss, :});          % Unpack data of current set-size group "ss"
            dat_type_ss = sortrows(dat_type_ss(:, :, 3), 1); % Select proportion saccades to stimulus of interest (not seperating for target/distractor)
            lck_sacc    = unique(dat_type_ss(:, 1));
            lck_sacc    = lck_sacc(~isnan(lck_sacc));
            no_lck      = numel(lck_sacc);
            for lck = 1:no_lck % Saccade in sequence

                for s = 1:no_valSubs % Subject

                    dat_singleSub(s, lck, ss, dt) = dat_type_pad{ss, s}(lck, 4);

                end

            end

%             % Exclude datapoints with too little subjects
%             li_excld = sum(~isnan(dat_singleSub(:, :, ss, dt)), 1) < min_sub;
% 
%             dat_singleSub(:, li_excld, ss, dt) = NaN;

        end

    end

end