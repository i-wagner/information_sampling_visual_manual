function gazeTraceFormatted = formatGazeTrace(gazeTrace)

    % Formats output of the "getGazeTrace" function, so that it only stores
    % data from whichever eye was actually recorded.
    %
    % NOTE
    % right now, this function cannot deal with cases when both eye where
    % recorded
    %
    % Input
    % gazeTrace: matrix; gaze trace, as returned by the "getGazeTrace"
    % function
    %
    % Output
    % gazeTraceFormatted: matrix; same as input, but formated so that it
    % only contains the data from whichever eye was actually recorded

    %% Formate gaze trace
    % Whenever the eye tracker cannot find the pupile, it assigns some
    % arbitrarily large (or small, rather) value as dummy value.
    flagDataloss = -32768;

    % Depending on which playback routine was used to save the gaze trace,
    % the resulting .dat file might have a different number of columns. The
    % number of columns depends on things like the number of recorded eye
    % (one eye or both), or whether the pupil signal was stored or not.
    gazeTraceFormatted = gazeTrace;
    nColumns = size(gazeTrace, 2);
    if nColumns > 4 % File contains data from both eyes
        if nColumns == 5 % Data of one eye + pupil signal
            error("Could not determine which eye's data was " + ...
                  "recorded please check 'formatGazeTrace' function.");
            gazeTraceFormatted(:, 4:5) = fliplr(gazeTraceFormatted(:, 4:5));
        elseif nColumns == 9
            % Reformat the .dat file so that it only contains data from
            % whichever eye was recorded. Columns of the non-recorded eye
            % only contain dummy values
            rightEyeRecorded = all(gazeTraceFormatted(:,2:3) == flagDataloss);
            leftEyeRecorded = all(gazeTraceFormatted(:,6:7) == flagDataloss);
            if rightEyeRecorded
                gazeTraceFormatted(:,2:3) = gazeTraceFormatted(:,6:7);
                gazeTraceFormatted(:,4) = gazeTraceFormatted(:,5) + ...
                                          gazeTraceFormatted(:,9);
                gazeTraceFormatted(:,5) = gazeTraceFormatted(:,8);
                gazeTraceFormatted(:,6:end) = [];
            elseif leftEyeRecorded
                gazeTraceFormatted(:,5) = gazeTraceFormatted(:,5) + ...
                                          gazeTraceFormatted(:,9);
                gazeTraceFormatted(:,4:5) = fliplr(gazeTraceFormatted(:,4:5));
                gazeTraceFormatted(:,6:end) = [];
            else
                error("Could not determine which eye's data was " + ...
                      "recorded please check 'formatGazeTrace' function.");
            end
        else
            error("Could not determine which eye's data was " + ...
                  "recorded please check 'formatGazeTrace' function.");
        end
    end

end