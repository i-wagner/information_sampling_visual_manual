function adjustedCoordinates = adjustVerticalCoordinates(coordinates, referencePoint)

    % Re-center coordinates around some reference point
    %
    % NOTE:
    % this function is intended to only correct coordinates of one axis.
    % Multiple axis, passed simultaneously, might or might not work
    % (untested)
    %
    % Input
    % coordinates:
    % arbitrary format (can be matrix, vector, or singular value);
    % coordinates to correct
    %
    % referencePoint:
    % integer; refrence point around which to center coordinates
    %
    % Output
    % adjustedCoordinates:
    % arbitrary format; same as "coordinates", but recentered

    %% Adjust coordinates
    adjustedCoordinates = coordinates - referencePoint;

end