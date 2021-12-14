function [mean_dt, single_dt] = infSampling_avgPropSacc(dat_input, min_sub)

    % Calculate mean proportion gaze shifts to stimulus category of
    % interest for a sequence of saccades
    % Input
    % dat_input: cell array with rows being single-subject data and
    %            columns different data types (for example, locked to
    %            last saccade in trial/trial start or proportion
    %            saccades to chosen/easy set). Each cells contains
    %            another cell array, with rows being individual
    %            set-size groups, with each of those cells containing
    %            the actuall data:
    %            (:, 1): lock index
    %            (:, 2): overall number saccades for this lock index
    %            (:, 3): number saccades to set
    %            (:, 4): proportion saccades to set
    %            (:, 4): proportion saccades not to set
    %            Pages correspond to saccades to target (:, :, 1), to
    %            distractor (:, :, 2) and to set (:, :, 3)
    % min_sub:   minimum number participants required to calcualte
    %            mean; datapoints with less subjects are excluded
    % Output
    % mean_dt:   cell array, with rows being the individual data
    %            types, provided as input. Each cell contains another
    %            cells array, with each row containing the aggregated
    %            data for each set-size group, provided with the
    %            input:
    %            (:, 1): lock index
    %            (:, 2): mean proportion saccades to set
    %            (:, 3): confidence interval of mean
    %            (:, 4): number saccades to set
    % single_dt: proportion gaze shifts on AOI for individual gaze shift of
    %            individual subjects; rows are data from subjects, columns
    %            are different gaze shifts in trials and pages are
    %            different data types (gaze shifts to chosen target, etc.)

    %% Get max. number of gaze shifts and numbered of subjects with data
    all_dat       = horzcat(dat_input{:});
    all_dat_maxGs = cellfun(@max, all_dat, 'UniformOutput', false);
    all_dat_maxGs = vertcat(all_dat_maxGs{:});
    all_dat_maxGs = max(all_dat_maxGs(:, :, 3));
    all_dat_subs  = max(sum(cell2mat(cellfun(@(x) size(x, 1), dat_input, 'UniformOutput', false))));


    %% For each set-size group, calculate mean proportion saccades to set
    no_dt     = size(dat_input, 2);
    mean_dt   = cell(no_dt, 1);
    single_dt = NaN(all_dat_subs, all_dat_maxGs(1), no_dt);
    for dt = 1:no_dt % Data type

        % Unpack set-size-groups data
        dat_single     = dat_input(:, dt);
        dat_single     = horzcat(dat_single{:});
        dat_single_pad = cellfun(@(x) x(:, :, 3), dat_single, 'UniformOutput', false);
        dat_single_pad = padCells(dat_single_pad, 20);

        % Average over timecourse for each set-size
        no_ss         = size(dat_single, 1); % Number set-size groups
        mean_ssGroups = cell(no_ss, 1);
        for ss = 1:no_ss % Set-size groups

            dat_single_ss = vertcat(dat_single{ss, :});                          % Unpack data of current set-size group "ss"
            dat_single_ss = sortrows(dat_single_ss(:, :, 3), 1);                 % Select proportion saccades to set (not seperating for target/distractor)

            % Calculate mean and CI for each datapoint in timecourse
            lck_sacc = unique(dat_single_ss(:, 1));
            lck_sacc = lck_sacc(~isnan(lck_sacc));
            no_lck   = numel(lck_sacc);
            mean_ss  = NaN(no_lck, 4);
            for lck = 1:no_lck

                li_dat_lock     = dat_single_ss(:, 1) == lck_sacc(lck);
                mean_ss(lck, 1) = lck_sacc(lck);
                mean_ss(lck, 2) = nanmean(dat_single_ss(li_dat_lock, 4)); % Mean
                mean_ss(lck, 3) = ci_mean(dat_single_ss(li_dat_lock, 4)); % Confidence intervals
                mean_ss(lck, 4) = nansum(li_dat_lock);                    % Number subjects
                for s = 1:all_dat_subs

                    single_dt(s, lck, dt) = dat_single_pad{s}(lck, 4);

                end

            end
            single_dt(:, mean_ss(:, 4) < min_sub, dt) = NaN;
            mean_ss(mean_ss(:, 4) < min_sub, :)       = [];                    % Exclude datapoints with too little subjects

            % Store set-size means
            mean_ssGroups{ss} = mean_ss;

        end

        % Store means for plots
        mean_dt{dt} = mean_ssGroups;

    end

end