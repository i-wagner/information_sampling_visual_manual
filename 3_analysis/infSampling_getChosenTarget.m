function [chosenTarget_resp, chosenTarget_fix, li_congruence] = ...
            infSampling_getChosenTarget(gapLocations_easy, response, fix_aoi, flag_targ, flag_dis, flag_bg)

    % Determine which target was chosen by subject
    % Input
    % gapLocationsEasy: column-vector with the gap-positions (up, down,
    %                   left, right) of the easy target
    % response:         column-vector with response in trial
    % fix_aoi:          column-vector, with flags of each fixated AOI in a
    %                   trial
    % flag_targ:        flags, marking a target
    % flag_dis:         flags, marking a distractor
    % flag_bg:          flag, marking the background
    % Output
    % chosenTarget_resp: chosen target (1: easy, 2:hard), determined based
    %                    on the response given by subject
    % chosenTarget_fix:  chosen target (1: easy, 2:hard), determined based
    %                    on last fixated target
    % li_congruence:     logical index, flagging if the last stimulus a
    %                    participant fixated corresponds to the target on
    %                    which a participant responded

    %% Determine chosen target based on response
    % The chosen target is defined as the target a participant responded on 
    % (e.g., if the easy target had the gap at its upper/lower side and
    % participant pressed the up/down button, we interpret this as a choice
    % for the easy target)
    if response == 1 | response == 2     % Down/up

        % If participant reported the gap location as being "up" or "down",
        % check if the gap of the easy target was positioned at either one
        % of those positions; if it was, a participant chose to respond on
        % the easy target, otherwise the participant chose the hard target
        if gapLocations_easy == 1 | gapLocations_easy == 2

            chosenTarget_resp = 1;

        else

            chosenTarget_resp = 2;

        end

    elseif response == 3 | response == 4 % Left/right

        % If participant reported the gap location as being "left" or
        % "right", check if the gap of the easy target was positioned at
        % either one of those positions; if it was, a participant chose to
        % respond on the easy target, otherwise the participant chose the
        % hard target
        if gapLocations_easy(1) == 3 | gapLocations_easy(1) == 4

            chosenTarget_resp = 1;

        else

            chosenTarget_resp = 2;

        end

    end


    %% Determine chosen target based on last fixated AOI
    % If the last gaze shift in a trial landed on any target, this target
    % is defined as the chosen target. If the last gaze shift landed on the
    % background, but the second-to-last gaze shift landed on any target,
    % this target is defined as the chosen target. If the last gaze shift
    % landed on any distractor, no target was chosen
    no_gs            = numel(fix_aoi);
    chosenTarget_fix = NaN;
    if no_gs > 1      % More than one gaze shift in trial
        
        stim_lastGs = fix_aoi(end-1:end);
        if     any(stim_lastGs(2) == flag_targ) % Last gaze shift on any target
            
            chosenTarget_fix = stim_lastGs(2);
            
        elseif stim_lastGs(2) == flag_bg && ... % Last gaze shift to background, second-to-last on any target
               any(stim_lastGs(1) == flag_targ)
            
            chosenTarget_fix = stim_lastGs(1);
            
        end
        
    elseif no_gs == 1 % One gaze shift in trial
        
        stim_lastGs = fix_aoi(end);
        if     any(stim_lastGs == flag_targ)       % Last gaze shift on any target
            
            chosenTarget_fix = stim_lastGs;
            
        end

    end


    %% Check if responded and last fixated target correspond
    % We also evaluate trials in which participants did fixate a
    % distractor/the background before giving their response
    li_congruence = chosenTarget_fix == chosenTarget_resp;

end