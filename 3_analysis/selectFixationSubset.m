function subset = selectFixationSubset(fixatedAois, idBg)

    % Wrapper function to select subset of fixations
    % Subsets are selected based on the following criteria
    % - Only unique fixations, i.e., no repeated fixations within the same
    %   AOI, without leaving the AOI first
    % - Exclude last fixation if it landed outside any AOI
    %
    % Input
    % fixatedAois:
    % vector; UNIQUE IDs of fixated AOIs. CAREFUL: don't use group IDs,
    % since those are, by defnition, not unique, and don't allow to extract
    % unique AOI fixations
    % 
    % idBg:
    % integer; flag for background fixations
    %
    % Output
    % subset:
    % vector; fixation selected for subset?

    %% Get unique AOI fixations 
    isUnique = getUniqueFix(fixatedAois);

    %% Check whether last unique gaze shift landed on the background
    uniqueFixations = fixatedAois(isUnique);

    lastOnBackground = ...
        checkLastGazeShift(uniqueFixations, idBg);

    %% Select subset
    idxLastUnique = find(isUnique, 1, 'last');

    subset = isUnique;
    subset(idxLastUnique) = ~lastOnBackground;

end