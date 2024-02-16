function gazeTraceSanitized = checkGazeSamples(gazeTrace, screenLimits)

    % Check gaze traces for implausible gaze-position-values. This includes
    % samples with datasloss, which where however not flagged as such, or
    % gaze samples where participants fixated outside of screen bounds.
    %
    % NOTE 1
    % This function taked a formated gaze trace, as returned by the
    % "formatGazeTrace" function, as input.
    %
    % NOTE 2
    % Flagging occurs by flagging corresponding gaze samples as dataloss
    %
    % Input
    % gazeTrace: matrix; gaze trace, as returned by the "formatGazeTrace"
    % function
    % screenLimits: vector; x and y limits of the screen that was used when
    % the gaze traces were recorded. Has to be pixel values.
    %
    % Output
    % gazeTraceSanitized: matrix; same as inpit, but with implausible
    % gaze samples flagged as such

    %% Check for not-detected blinks/data-loss
    % We check if there is any datasample with coordiantes outside the
    % measurable screen area and if, for this sample, the bit for a blink/a
    % saccade is not turned on. If this is the case, we turn the bit on so
    % we can later detect the blink/dataloss
    notDataloss = bitget(gazeTrace(:,4), 2) == 0;
    notSaccade = bitget(gazeTrace(:,4), 1) == 0;
    isOutOfBounds = ...
        (gazeTrace(:,2) > screenLimits(1) & notDataloss & notSaccade) | ...
        (gazeTrace(:,2) < 0 & notDataloss & notSaccade) | ...
        (gazeTrace(:,3) > screenLimits(2) & notDataloss & notSaccade) | ...
        (gazeTrace(:,3) < 0 & notDataloss & notSaccade);

    gazeTraceSanitized = gazeTrace;
    gazeTraceSanitized(isOutOfBounds,4) = ...
        bitset(gazeTraceSanitized(isOutOfBounds,4), 2);

end