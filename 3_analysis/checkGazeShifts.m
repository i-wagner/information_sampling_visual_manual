function exclude = checkGazeShifts(coordOn, coordOff, meanGazePos, duration, screenBounds, minDur)

    % Checks if gaze shifts satisfy exclusion criteria
    % Three exclusion criteria are checked by this function:
    % - Gaze shift is very short. 
    %   Very short gaze shifts sometimes occur after dataloss; we treat
    %   those very short gaze shifts as false-positives and exclude them 
    %   later
    % - Gaze shift is missing an on- or offset.
    %   Incomplete gaze shifts sometimes occur if a participant decides to 
    %   give a response while a gaze shift is still in flight
    % - Gaze had on-/offset-coordinates outside of screen bounds
    %   Gaze shifts with on-/offset coordinates outside the screen 
    %   sometimes occur if participants, for example, check the keyboard
    %   keys before placing a response.
    %
    % Input
    % coordOn: 
    % matrix; gaze coordinates (vertical and horizontal) at gaze shift 
    % onset 
    %
    % coordOff:
    % matrix; gaze coordinates (vertical and horizontal) at gaze shift 
    % offset 
    %
    % meanGazePos:
    % matrix; mean gaze position (vertical and horizontal) inbetween gaze
    % shifts
    %
    % duration:
    % vector; duration of gaze shifts
    %
    % screenBounds:
    % structure with fields .X and .Y;
    %
    % minDur:
    % integer; minimum duration for gaze shifts
    %
    % NOTE:
    % gaze coordinates and gaze positions have to be matrices, with x- and
    % y-coordinates stored in the columns. The first column must be x, the
    % second y
    %
    % NOTE 2:
    % all values must be provided in dva
    %
    % NOTE 3:
    % screenBounds.Y must contain two values, one for the lowr and one for
    % the upper screen bound. The fixation cross in this experiment was not 
    % placed at the vertical screen center, but at the lower screen border. 
    % Thus, the area around the fixation cross was not symetrical, and we 
    % cannot just use one value to check whether participants looked at a
    % weird location
    %
    % Output
    % exclude:
    % vector; Booleans, indicating whether the gaze shift has to be
    % excluded

    %% Perform checks
    horCoord = [coordOn(:,1), coordOff(:,1), meanGazePos(:,1)];
    vertCoord = [coordOn(:,2), coordOff(:,2), meanGazePos(:,2)];

    isShort = duration < minDur;
    offsetMissing = any(isnan(coordOff), 2);
    outOfBoundsHor = any(abs(horCoord) > screenBounds.X, 2);
    outOfBoundsVert = any(vertCoord > screenBounds.Y(1), 2) | ...
                      any(vertCoord < screenBounds.Y(2), 2);

    exclude = isShort | offsetMissing | outOfBoundsHor | outOfBoundsVert;

end