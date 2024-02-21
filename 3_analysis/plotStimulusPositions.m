function plotStimulusPositions(horEndpoints, vertEndpoints, horStimCenter, vertStimCenter)

    % Creates a visualisation of the screen area, the stimulus area,
    % stimulus position and gaze shift endpoints
    %
    % Input
    % horEndpoints: 
    % vector; horizontal coordinates of gaze shift endpoint
    %
    % vertEndpoints: 
    % vector; vertical coordinates of gaze shift endpoint
    %
    % horStimCoord: 
    % vector; horizontal coordinates of stimuli
    %
    % vertStimCoord: 
    % vector; vertical coordinates of stimuli
    %
    % Output
    % --

    %% Define some parameters
    % Size of one individual stimulus
    STIM_DIAMETER = 1.2512;
    STIM_RADIUS = STIM_DIAMETER / 2;
    AOI_RADIUS = 1.50;

    % Fixation cross location and fixation tolerance
    FIX_TOL_RADIUS = AOI_RADIUS;
    FIX_LOC = [0, -9.50];

    % On-screen area, within which stimuli are presented
    horCoordStimulusArea = [-8, -8, 8, 8, -8];
    vertCoordStimulusArea = [(FIX_LOC(2) + 4), (FIX_LOC(2) + 20), ...
                             (FIX_LOC(2) + 20), (FIX_LOC(2) + 4), ...
                             (FIX_LOC(2) + 4)];

    % Right now the vertical position of stimuli and gaze shifts is aligned
    % with the fixation cross. To make plotting easier, we center stimuli
    % on zero (the vertical screen center)
    vertStimCenter = vertStimCenter + FIX_LOC(2);
    vertEndpoints = vertEndpoints + FIX_LOC(2);

    %% Get coordinates of on-screen elements
    % Stimuli and AOIs
    unitCircle = 0:0.01:2*pi;
    aoiArea = [cos(unitCircle) .* AOI_RADIUS; ...
               sin(unitCircle) .* AOI_RADIUS];
    stimArea = [cos(unitCircle) .* STIM_RADIUS; ...
                sin(unitCircle) .* STIM_RADIUS];
    nStimuli = numel(horStimCenter);
    horCoordStim = [];
    vertCoordStim = [];
    for s = 1:nStimuli % Stimulus
        % Define area, occupied by areas of interest. Add AOIs to the
        % vector with the position of the stimulus area, so we can check
        % whether a gaze shift landed on the stimulus area or any aoi
        horCoordStimulusArea  = [horCoordStimulusArea, ...
                                 NaN, ...
                                 (aoiArea(1,:) + horStimCenter(s))];
        vertCoordStimulusArea  = [vertCoordStimulusArea, ...
                                  NaN, ...
                                  (aoiArea(2,:) + vertStimCenter(s))];

        % Define area, occupied by stimuli
        horCoordStim = [horCoordStim, ...
                        (stimArea(1,:) + horStimCenter(s)), ...
                        NaN];
        vertCoordStim = [vertCoordStim, ...
                         (stimArea(2,:) + vertStimCenter(s)), ...
                         NaN];
    end

    % Check which gaze shift offset ended somewhere on the screen,
    % outside any AOI
    [horCoordStimulusArea, vertCoordStimulusArea] = ...
        poly2cw(horCoordStimulusArea, vertCoordStimulusArea);
    gsInAoi = inpolygons(horEndpoints, vertEndpoints, ...
                         horCoordStimulusArea, ...
                         vertCoordStimulusArea);

    %% Create plot
    horCoordFixTol = cos(unitCircle) .* FIX_TOL_RADIUS + FIX_LOC(1);
    vertCoordFixTol = sin(unitCircle) .* FIX_TOL_RADIUS + FIX_LOC(2);

    clf
    [f, v] = poly2fv(horCoordStimulusArea, vertCoordStimulusArea);
    patch('Faces', f, ...
          'Vertices', v, ...
          'FaceColor', [0.90, 0.90, 0.90], ...
          'EdgeColor', 'none');
    hold on
    plot(horStimCenter(~isnan(horStimCenter)), ...
         vertStimCenter(~isnan(vertStimCenter)), ...
         '*g', ...
         'MarkerSize', 10);
    plot(horCoordStim, vertCoordStim);

    plot(FIX_LOC(1), FIX_LOC(2), ...
         '+', ...
         'MarkerSize', 15);
    plot(horCoordFixTol, vertCoordFixTol);
    plot(horEndpoints(gsInAoi), vertEndpoints(gsInAoi), 'r.', ...
         horEndpoints(~gsInAoi), vertEndpoints(~gsInAoi), 'b.', ...
         'MarkerSize', 15);
    text(horEndpoints + 0.30, vertEndpoints + 0.30, ...
         num2str((1:numel(horEndpoints))'));
    hold off
    axis([-24.5129, 24.5129, -13.7834, FIX_LOC(2) + 10.50]);
    daspect([1, 1, 1]);
    disp("Please press a button to continue!");
    waitforbuttonpress

end