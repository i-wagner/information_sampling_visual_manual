function [allProb, allFix] = recursiveProb(bias, setSize, BIASFIRSTFIX, remainingProb, fixCount, allProb, allFix)

    %RECURSIVEMODEL Summary of this function goes here
    %   Detailed explanation goes here

    %% Init
    if nargin < 7

        remainingProb = 1;     % remaining probability to find target
        fixCount      = [0 0]; % # fixations made
        for s = 1:2 % Set

            allProb{s} = [];
            for f = 1:2 % Fixation

                allFix{s, f} = [];

            end

        end

    end


    %% Bias first fixation?
    UNBIASED_FIRST_FIXATION = 1;
    if BIASFIRSTFIX == 1

        UNBIASED_FIRST_FIXATION = 0;

    end


    %% Make predictions
    for s = 1:2 % Set

        % Probability to choose element from set
        if UNBIASED_FIRST_FIXATION && sum(fixCount) == 0                                            % First fixation is unbiased

            keyboard
            sel(s) = setSize(s) ./ (setSize(1) + setSize(2));                           

        else

            if bias < 1                                                                             % Easy set prefered

                if s == 1                                                                           % Easy set, when easy set prefered

                    sel(s) = setSize(s) ./ (setSize(1) + (bias .* setSize(2)));

                else                                                                                % Difficult set, when easy set prefered

                    sel(s) = (bias .* setSize(s)) ./ (setSize(1) + (bias .* setSize(2)));

                end

            else                                                                                    % Difficulty set prefered

                if s == 1                                                                           % Easy set, when difficulty set prefered

                    sel(s) = ((2 - bias) .* setSize(s)) ./ ((2 - bias) .* setSize(1) + setSize(2));

                else                                                                                 % Difficult set, when difficult set prefered

                    sel(s) = setSize(s) ./ ((2 - bias) .* setSize(1) + setSize(2));

                end

            end

        end

        % Calculate probabilities to find/miss target with the current fixation
        % Hit  (i.e., probability to fixate set and find target):     probability_selectElementFromSet * probability_findTargetInSet     * probability_targetNotFound
        % Miss (i.e., probability to fixate set and find distractor): probability_selectElementFromSet * probability_findDistractorInSet * probability_targetNotFound
        hitProbability(s)  = sel(s) .* (1 / setSize(s))                 .* remainingProb; % Probability to find target in set ( P(findTarget | targetNotFoundYet )
        missProbability(s) = sel(s) .* ((setSize(s) - 1) ./ setSize(s)) .* remainingProb; % Probability to miss target in set ( P(findTarget | targetNotFoundYet )

        % Store finished path when target has been found
        allProb{s}     = [allProb{s}     hitProbability(s)]; % Probability to find target for each fixation
        allFix{s, s}   = [allFix{s, s}   fixCount(s)+1];     % # fixation to the current set
        allFix{s, 3-s} = [allFix{s, 3-s} fixCount(3-s)];     % # fixations to the other set

        % Continue path if more than one element is left in set
        if setSize(s) > 1

            tmp_setSize       = setSize;            % # remaining elements in current set "s"
            tmp_setSize(s)    = tmp_setSize(s) - 1;
            tmp_fix           = fixCount;           % # fixations made
            tmp_fix(s)        = tmp_fix(s) + 1;
            [allProb, allFix] = recursiveProb(bias, tmp_setSize, BIASFIRSTFIX, missProbability(s), tmp_fix, allProb, allFix);

        end

    end

end