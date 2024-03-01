function [subset, passedQualityCheck] = ...
    selectFixationSubset(fixatedAois, tsGsOnset, tsGsOffset, horCoord, vertCoord, gsDuration, tsStimOnset, tsResponse, idBg, screenBounds, minDur)

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
    % tsGsOnset:
    % vector; timestamps of gaze shift onsets
    % 
    % tsGsOffset:
    % vector; timestamps of gaze shift offsets
    % 
    % horCoord: 
    % matrix; horizontal coordinates of gaze to check for out-of-bounds
    % samples. Can have an arbitrary number columns, each representing some
    % coordinate (e.g., onset and offset coordinates)
    % 
    % vertCoord: 
    % matrix; vertical coordinates of gaze to check for out-of-bounds
    % samples. Can have an arbitrary number columns, each representing some
    % coordinate (e.g., onset and offset coordinates)
    % 
    % gsDuration:
    % vector; duration of each gaze shift
    % 
    % tsStimOnset: 
    % integer; timestamp of stimulus onset
    % 
    % tsResponse:
    % integer; timestamp of stimulus offset
    % 
    % idBg:
    % integer; flag for background fixations
    %
    % screenBounds:
    % structure with fields X and Y; screen bounds for each screen
    % dimension. "X" has only one value, because the fixation cross was at
    % horizontal screen center. "Y" needs two values, to account for the
    % fact that the fixaiton cross was not equidistant to the upper/lower
    % screen edge
    %
    % minDur:
    % integer; minimum duration of gaze shifts
    %
    % Output
    % subset:
    % vector; fixation selected for subset?

    %% First, perform some quality checks
    isShort = gsDuration < minDur;
    offsetMissing = any(isnan([horCoord, vertCoord]), 2);
    offsetAfterResponse = tsGsOffset > tsResponse;
    outOfBoundsHor = any(abs(horCoord) > screenBounds.X, 2);
    outOfBoundsVert = any(vertCoord > screenBounds.Y(1), 2) | ...
                      any(vertCoord < screenBounds.Y(2), 2);

    passedQualityCheck = ~isShort & ~offsetMissing & ~outOfBoundsHor & ...
                         ~outOfBoundsVert & ~offsetAfterResponse;

    %% Second, select fixations based of inclusion criteria
    % Only get unique fixations after stimulus onset. This bit of code is
    % not directly necessary. It was only included to replicate the results
    % from the old pipeline (i.e., for sanity checks)
    subset = (tsGsOnset >= tsStimOnset) & passedQualityCheck;

    % Get unique AOI fixations 
    subset(subset) = getUniqueFix(fixatedAois(subset));

    % Check whether last unique gaze shift landed on the background
    % Only do this for cases where participants actually made any agze
    % shifts to something other than the background (or made any gaze
    % shifts at all)
    if ~all(~subset)
        lastOnBackground = ...
            checkLastGazeShift(fixatedAois(subset), idBg);
    
        idxLastUnique = find(subset, 1, 'last');
        subset(idxLastUnique) = ~lastOnBackground;
    end

end