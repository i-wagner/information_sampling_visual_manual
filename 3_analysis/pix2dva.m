function samplesDva = pix2dva(samples, screenCenter, pix2deg, flipSign)

    % Converts pixel coordinates to degrees of visual angle. The converted
    % value is aligned to the screen center, and corresponds to the
    % degree-distance to it.
    %
    % NOTE:
    % The function expects one set of values, either vertical or horizontal
    % coordinates, as input. If both need to be converted, the function has
    % to be called twice with the corresponding parameter.
    %
    % Input
    % samples: vector; samples with values to convert
    % screenCenter; integer; pixel coordinates of screen center
    % pix2deg; float; conversion factor 
    % flipSign; boolean; flip sign of samples?
    %
    % Output
    % samplesDva: vector; same as input, but converted to dva

    %% Convert gaze coordinates from pixel to degrees of visual angle
    samplesDva = (samples - screenCenter) .* pix2deg;
    if flipSign
        samplesDva = samplesDva .* -1;
    end

end