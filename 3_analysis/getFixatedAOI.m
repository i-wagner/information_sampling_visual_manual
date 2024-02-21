function aoiVisits = getFixatedAOI(horEndpoints, vertEndpoints, horStimCoord, vertStimCoord, aoiRadius, bgFlag)

    % Determines if a gaze shift landed within the AOI around one of the
    % shown stimuli, and if yes, which stimulus was targeted
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
    % aoiRadius: 
    % float; radius of area of interest
    %
    % bgFlag: 
    % integer; flag used to mark gaze shift that landed on the background, 
    % i.e., outside any defined AOI
    %
    % Output
    % aoiVisits: 
    % vector; indices of stimuli that were targeted by gaze 
    % shift. Indices correspond to the location of the stimulus in the
    % stimulus coordinate vectors. Gaze shifts that landed outside any AOI
    % will get the "background" flag

    %% Get fixated AOIs
    % We define a circular AOI with a fixed radius around each stimulus.
    % Predefine the AOI so we don't have to re-calculate the area in each
    % loop iteration
    unitCircle = 0:0.01:2*pi;
    aoiArea = [cos(unitCircle) .* aoiRadius; ...
               sin(unitCircle) .* aoiRadius];

    % Check for each gaze shift whether it's endpoint falls within the AOI
    % around any of the shown stimuli
    nGazeShifts = numel(horEndpoints);
    nStimuli = numel(horStimCoord);
    aoiVisits = NaN(nGazeShifts, nStimuli);
    for g = 1:nGazeShifts % Gaze shift
        for s = 1:nStimuli % Stimulus
            aoiCoordHor = aoiArea(1,:) + horStimCoord(s);
            aoiCoordVert = aoiArea(2,:) + vertStimCoord(s);
            gsInAoi = inpolygon(horEndpoints(g), vertEndpoints(g), ...
                                aoiCoordHor, aoiCoordVert);
            if gsInAoi
                aoiVisits(g,s) = s;
            end
        end
    end
    aoiVisits = sum(aoiVisits, 2, 'omitnan');

    % A gaze shift landed on the background if it has both, an onset and an 
    % offset, and if it did not land on any of the defined AOIs
    gsOnBackground = (aoiVisits == 0) & ...
                     ~isnan(horEndpoints) & ...
                     ~isnan(vertEndpoints);
    aoiVisits(gsOnBackground) = bgFlag;
    aoiVisits(aoiVisits == 0) = NaN; % Gaze shift without offset

end