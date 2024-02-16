function [gazeTrace, dataLoss] = getGazeTrace(thisSubject, thisCondition, thisTrial, pathData, screen)

    % Wrapper function to load an process a subjectgaze trace.
    %
    % The following steps are performed:
    % - Load the fiel with the gaze trace
    % - Format the gaze trace so it only contains values from the recorded eye
    % - Check gaze trace for sample integrity
    % - Convert gaze coordinates from pixel to dva.
    %   NOTE: y-traces are flipped, so that positive values correspond to
    %         positions in the screen half above the vertical screen center
    %
    % Input
    % thisSubject: integer; ID of current subject
    % thisCondition: integer; ID of current condition
    % thisTrial: integer; ID of current trial
    % pathData: string; path to data folder
    % screen: structure; variable with parameters of screen, on which the
    % experiment was conducted
    %
    % Output
    % gazeTrace: matrix; gaze trace of participant
    % dataLoss: boolean; samples missing from gaze trace?
    %
    % NOTE:
    % gazeTrace has the following columns:
    % - Timestamps when sample was recorded
    % - x-coordinate of gaze (dva)
    % - y-coordinate of gaze (dva)
    % - bit-flag, which marks eye-link events in the trial (i.e., event
    %   trigger, sent by eye-link)
    % - pupil size (pixel)

    %% Get gaze trace
    screenSize = [screen.size.x.PX, screen.size.y.PX];

    [gazeTrace, dataLoss] = ...
        loadDat(thisSubject, thisCondition, thisTrial, pathData);
    gazeTrace = formatGazeTrace(gazeTrace);
    gazeTrace = checkGazeSamples(gazeTrace, screenSize);
    gazeTrace(:,2) = pix2dva(gazeTrace(:,2), screen.center.x.PX, ...
                             screen.pix2deg.X, false);
    gazeTrace(:,3) = pix2dva(gazeTrace(:,3), screen.center.y.PX, ...
                             screen.pix2deg.Y, true);

end