function nDistractorsChosenSet = getDistractorsChosenSet(nDistractors, chosenTargetId, targetId)

    % Get number of distractors in stimulus set of the chosen target
    %
    % Input
    % nDistractors:
    % vector; number if easy and difficult distractors. NUMBER OF EASY
    % DISTRACTORS HAS TO COME FIRST
    %
    % chosenTargetId:
    % integer; ID of chosen target
    %
    % targetId:
    % vector; IDs of targets
    %
    % Output
    % nDistractorsChosenSet:
    % integer; number of distractors in stimulus set of the chosen target

    %% Get number of distractors in stimulus set of the chosen target
    if chosenTargetId == targetId(1)
        nDistractorsChosenSet = nDistractors(1);
    elseif chosenTargetId == targetId(2)
        nDistractorsChosenSet = nDistractors(2);
    else
        error("Invalid target ID!");
    end

end
