function no_unique_aoi_fixated = infSampling_getUniqueFixations(fixated_aoi, id_targ, id_bg, curr_cond)

    % Count how many stimuli very fixated
    % Input
    % fixated_aoi:           vector with unique identifier of each stimulus,
    %                        fixated in a trial (including fixations of
    %                        background)
    % id_targ:               vector with unique identifier of "target"
    %                        stimuli
    % id_bg:                 scaler of unique identifier of fixations
    %                        outside any defined AOI (i.e., gaze shifts on
    %                        background)
    % curr_cond:             origin of data (2 == single-target; 3 ==
    %                        double-target)
    % Output
    % no_unique_aoi_fixated: scaler representing the number of unique
    %                        stimuli, fixated during search

    %% Get which unique stimuli where fixated in a trial
    no_aoi_fixated     = numel(fixated_aoi);
    unique_aoi_fixated = NaN(no_aoi_fixated, 1);
    for a = 1:no_aoi_fixated % AOI

        curr_aoi = fixated_aoi(a);
        if ~ismember(curr_aoi, unique_aoi_fixated)

            unique_aoi_fixated(a) = curr_aoi;

        end

    end


    %% Prepare data
    % Remove fixations on background
    li_bg = unique_aoi_fixated == id_bg;

    unique_aoi_fixated(li_bg) = NaN;

    % Remove fixations on targets
    % In single-target, we remove every gaze shift that landed on the
    % target. In double-target, we only remove cases where the last gaze
    % shift in a trial went to a target
    li_targ                     = any(unique_aoi_fixated == id_targ, 2);
    unique_aoi_fixated(li_targ) = NaN;
%     li_targ = [];
%     if curr_cond == 2
% 
%         li_targ = logical(sum(unique_aoi_fixated == id_targ, 2));
% 
%     elseif curr_cond == 3 & ~isempty(unique_aoi_fixated)
% 
%         li_targ = logical(sum(unique_aoi_fixated(end) == id_targ, 2));
% 
%     end
% 
%     if any(li_targ)
% 
%         unique_aoi_fixated(li_targ) = NaN;
% 
%     end

%     % Remove last fixation on target
%     % We do not consider the last gaze shift in a trial, which landed on a
%     % target, as part of the visual search, thus, it is not counted for the
%     % number of stimuli fixated during search
%     if numel(unique_aoi_fixated) > 0 && ...
%        any(unique_aoi_fixated(end) == id_targ)
% 
%         unique_aoi_fixated(end) = NaN;
% 
%     % Legacy analysis; technically redundant
%     % If we keep fixations on the background, this code-chunk can be used to
%     % exclude cases in which the last gaze shifte landed on the background,
%     % but the second-to-last gaze shift landed on one of the targets
%     elseif (numel(unique_aoi_fixated) > 1 && ...
%             any(unique_aoi_fixated(end-1) == id_targ) && ...
%             unique_aoi_fixated(end) == id_bg)
% 
%        keyboard
%        unique_aoi_fixated(end-1:end) = NaN;
% 
%     end


    %% Count how many unique stimuli were fixated
%     no_unique_aoi_fixated = [sum(~isnan(unique_aoi_fixated)) ...
%                              sum(unique_aoi_fixated == 1 | (unique_aoi_fixated > 2 & unique_aoi_fixated <= 10)) ...
%                              sum(unique_aoi_fixated == 2 | (unique_aoi_fixated > 2 & unique_aoi_fixated > 10))];
    no_unique_aoi_fixated = [sum(~isnan(unique_aoi_fixated)) ...
                             sum((unique_aoi_fixated > 2 & unique_aoi_fixated <= 10)) ...
                             sum((unique_aoi_fixated > 2 & unique_aoi_fixated > 10))];
%     if no_unique_aoi_fixated == 0
% 
%         no_unique_aoi_fixated = NaN;
% 
%     end

end