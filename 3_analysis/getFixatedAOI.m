function [uniqueIds, groupIds] = getFixatedAOI(horEndpoints, vertEndpoints, horStimCoord, vertStimCoord, aoiRadius, bgFlag, edFlag, ddFlag)

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
    % edFlag:
    % integer; flag used to mark gaze shift that landed on an easy
    % distractor
    %
    % ddFlag:
    % integer; flag used to mark gaze shift that landed on a difficult
    % distractor
    %
    % Output
    % aoiVisits: 
    % matrix; indices of stimuli that were targeted by gaze shift and 
    % unique identifier of stimulus group. Indices correspond to the 
    % location of the stimulus in the stimulus coordinate vectors. Gaze 
    % shifts that landed outside any AOI will get the "background" flag.
    % Easy/difficult distractor will get a group variable so we can easily
    % locate them

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
    uniqueIds = NaN(nGazeShifts, nStimuli);
    for g = 1:nGazeShifts % Gaze shift
        for s = 1:nStimuli % Stimulus
            aoiCoordHor = aoiArea(1,:) + horStimCoord(s);
            aoiCoordVert = aoiArea(2,:) + vertStimCoord(s);
            gsInAoi = inpolygon(horEndpoints(g), vertEndpoints(g), ...
                                aoiCoordHor, aoiCoordVert);
            if gsInAoi
                uniqueIds(g,s) = s;
            end
        end
    end
    uniqueIds = sum(uniqueIds, 2, 'omitnan');

    % A gaze shift landed on the background if it has both, an onset and an 
    % offset, and if it did not land on any of the defined AOIs
    gsOnBackground = (uniqueIds == 0) & ...
                     ~isnan(horEndpoints) & ...
                     ~isnan(vertEndpoints);
    uniqueIds(gsOnBackground) = bgFlag;
    uniqueIds(uniqueIds == 0) = NaN; % Gaze shift without offset

    %% Assign a group variable to each fixated AOI
    % Initially, stimulus in a trial has a unique identifier, which
    % corresponds to the index of it's position in the location matrix. In
    % addition to this, we assign a high-level identifier, which marks the
    % group a stimulus belong to, i.e., easy/difficult distractor
    fixatedEasyDistractor = uniqueIds > 2 & uniqueIds <= 10;
    fixatedDifficultDistractor = uniqueIds > 10 & uniqueIds <= 18;
    fixatedDistractor = ...
        sum([fixatedEasyDistractor, fixatedDifficultDistractor], 2);

    groupIds = zeros(nGazeShifts,1);
    groupIds(fixatedEasyDistractor) = edFlag;
    groupIds(fixatedDifficultDistractor) = ddFlag;
    groupIds(~fixatedDistractor) = uniqueIds(~fixatedDistractor);

end