function nDistractorsRecoded = recodeDistractorNumber(nDistractors, shownTarget, ids)

    % Recodes the logged number of distractors in the single-target
    % condition, so that the number of distractors for the NOT-SHOWN target
    % is NaN
    %
    % Input
    % nDistractors: matrix; trialwise number of distractors
    % shownTarget: vector; ID of target that was shown in a trial
    % ids: vector; IDs that mark an easy and difficult target
    %
    % NOTE FOR ALL APPLICABLE INPUTS: FIRST COLUMN MUST CONTAIN INFORMATION
    % ABOUT EASY TARGETS/DISTRACTORS, SECOND COLUMN ABOUT DIFFICULT
    % TARGETS/DISTRACTORS
    %
    % Output
    % nDistractorsRecoded: matrix; same as input, but with the number of
    % distractors in the non-shown set being recoded

    %% Get trials where one or the other target was shown
    easyShown = shownTarget == ids(1);
    difficultShown = shownTarget == ids(2);

    %% Recode trials
    nDistractorsRecoded = nDistractors;

    nDistractorsRecoded(difficultShown,1) = NaN;
    nDistractorsRecoded(easyShown,2) = NaN;

end