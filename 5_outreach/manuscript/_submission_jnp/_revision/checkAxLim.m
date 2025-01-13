function doesCutoff = checkAxLim(limits, values)
    
    % Checks whether values lie outside of chosen axis bounds
    %
    % Input
    % limits:
    % vector; axis limits 
    %
    % values:
    % matrix; datapoints we intend to plot
    %
    % Output
    % doesCutoff:
    % Boolean; any values outside of chosen axis limits?

    %% Check whether values are in bounds
    assert((numel(limits) >= 2) & (numel(limits) <= 4), ...
           "'limits' has to be either a four-element or two-element, " + ...
           "vector specifying limits for x- and y-axis!");
    doesCutoff = (min(limits) > min(values(:))) | ...
                 (max(limits) < max(values(:)));

end