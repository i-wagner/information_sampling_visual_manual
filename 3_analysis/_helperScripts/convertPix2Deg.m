function [xTrace_deg, yTrace_deg] = convertPix2Deg(xTrace, yTrace, screenCenter, pix2deg, recodeY)

    % Converts gaze position in pixel to gaze position in degrees of
    % visual angle
    % Input
    % *Trace:       column-vector with x- and y-coordinates of gaze trace
    %               over time (in pixel)
    % screenCenter: two-column-matrix, containing horizontal and vertical
    %               screen center in pixel
    % pix2deg:      conversion factor to transform x/y pixel coordinates to
    %               degrees of visual angle
    % recodeY:      indicator, if we want to flip the y-axis (so positive
    %               y-values correspond to upper screen half);
    %               1 == flip
    %               0 == not flip
    % Output
    % *Trace_deg:  gaze traces over time in degrees of visual angle

    %% Convert pixel to deg
    xTrace_deg = (xTrace - screenCenter(1)) .* pix2deg(1);
    if recodeY == 0

        yTrace_deg = (yTrace - screenCenter(2)) .* pix2deg(2);

    elseif recodeY == 1

        yTrace_deg = ((yTrace - screenCenter(2)) .* pix2deg(2)) .* -1;

    end

end