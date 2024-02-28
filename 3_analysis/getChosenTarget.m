function [chosenTargetResp, chosenTargetFix] = getChosenTarget(gapLocation, response, fixatedAoi, flagTarget, flagBg)

    % Determines which target a participant chose, based on their response
    % (resposne method) as well as their last fixation (fixation method)
    %
    % NOTE 1:
    % This function, by default, determines the chosen target based on both
    % methods. Toggle the output if not interested in any of the methods
    %
    % NOTE 2:
    % Mapping of output is the same as provided in the "flagTarget" input
    %
    % RESPONSE METHOD:
    % The chosen target is defined as the target a participant responded on 
    % (e.g., if the easy target had the gap at its upper/lower side and
    % participant pressed the up/down button, we interpret this as a choice
    % for the easy target)
    %
    % FIXATION METHOD:
    % The chosen target is defined as the target on which the last gaze 
    % shift landed on. If the last gaze shift landed on the background, but 
    % the second-to-last gaze shift landed on any target, this target is 
    % defined as the chosen target. If the last gaze shift landed on any 
    % distractor, no target was chosen
    %
    % Input
    % gapLocation:
    % vector; gap position on easy and difficult target. EASY TARGET HAS TO
    % BE STORED FIRST
    %
    % response:
    % integer; gap position, as reported by participant
    %
    % fixatedAoi:
    % vector; unique IDs of fixated AOIs
    %
    % flagTarget:
    % vector; IDs that identify a target fixation
    %
    % flagBg:
    % integer; ID that identifies a distractor fixation
    %
    % Output
    % chosenTargetResp:
    % integer; ID of chosen target, determined based on participants 
    % response.
    %
    % chosenTargetFix:
    % integer; ID of chosen target, determined based on which stimulus a
    % participant fixated last in a trial

    %% Determine chosen target based on response
    % Check whether the gap was located at at the same stimulus axis as
    % indicated by a participant's response. E.e., if the gap is at the
    % upper side of the stimulus, check whether the up or down button was
    % pressed
    if response == 1 | response == 2 % Down or up button pressed
        if gapLocation(1) == 1 | gapLocation(1) == 2
            chosenTargetResp = flagTarget(1);
        else
            chosenTargetResp = flagTarget(2);
        end
    elseif response == 3 | response == 4 % Left or right button pressed
        if gapLocation(1) == 3 | gapLocation(1) == 4
            chosenTargetResp = flagTarget(1);
        else
            chosenTargetResp = flagTarget(2);
        end
    end

    %% Determine chosen target based on last fixated AOI
    nGazeShifts = numel(fixatedAoi);
    chosenTargetFix = NaN;
    if nGazeShifts > 1
        lastTwoFixations = fixatedAoi(end-1:end);
        if any(lastTwoFixations(2) == flagTarget)
            chosenTargetFix = lastTwoFixations(2);
        elseif lastTwoFixations(2) == flagBg & ...
               any(lastTwoFixations(1) == flagTarget)
            chosenTargetFix = lastTwoFixations(1); 
        end
    elseif nGazeShifts == 1
        lastFixation = fixatedAoi(end);
        if any(lastFixation == flagTarget)
            chosenTargetFix = lastFixation;
        end
    end

end