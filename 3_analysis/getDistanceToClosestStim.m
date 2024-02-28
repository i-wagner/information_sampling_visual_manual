function [wentToClosest, wentNotToClosest, distanceToClosest] = ...
    getDistanceToClosestStim(fixatedAoi, horStimCoord, vertStimCoord, horOnsetCoord, vertOnsetCoord, flagBg, correctCurrent)

    % Determines whether a gaze shift went to the stimulus cloests to the
    % current fixation location or to a stimulus further away
    %
    % NOTE:
    % Gaze shifts to a further away stimulus are not necessarily the
    % inverse of gaze shifts to the closest stimulus, because gaze
    % shiftsncan also go to the background, which don't fall into either
    % category
    %
    % Input
    % fixatedAoi:
    % vector; IDs of fixated AOIs. HAS TO BE UNIQUE IDs, because the ID
    % are, at the same time, also indices to the stimulus coordinates in
    % the coordinates vectors
    %
    % horStimCoord:
    % vector; horizontal stimulus coordiates
    %
    % vertStimCoord:
    % vector; vertical stimulus coordiates
    %
    % horOnsetCoord:
    % vector; horizontal gaze coordinates coordiates at gaze shift offset
    %
    % vertOnsetCoord:
    % vector; vertical gaze coordinates coordiates at gaze shift offset
    %
    % flagBg;
    % integer; flag to identify background fixations
    %
    % correctCurrent;
    % Boolean; correct for currently fixated stimulus, when calculating
    % distances? If not, the distance to the currently fixated stimulus
    % will be calculated, which results in the currently fixated stimulus
    % always being the closest stimulus
    %
    % Output
    % wentToClosest:
    % vector; did gaze shift go to closest stimulus?
    %
    % wentNotToClosest:
    % vector; did gaze shift not go to closest stimulus?
    %
    % distanceToClosest:
    % vector; Euclidean distance to closest stimulus

    %% Check whether gaze shift went to the stimulus closest to fixation
    % At the start of a trial, the gaze is at the fixation location, for
    % which we have no AOI; so we mark it as NaN. We don't check the last
    % fixation, because, by definition, we cannot calculate whether gaze
    % went to the closest stimulus for the fixation after the last one
    % (since there is none)
    currentAoi = [NaN; fixatedAoi(1:end-1)];
    nextAoi = fixatedAoi;
    nFixations = numel(currentAoi);
    nStimuli = numel(horStimCoord);

    wentToClosest = NaN(nFixations, 1);
    wentNotToClosest = NaN(nFixations, 1);
    distanceToClosest = NaN(nFixations, 1);
    for g = 1:nFixations % Gaze shift
        % Get distance between current fixation location and each stimulus
        % on the screen
        distanceToStimulus = NaN(nStimuli, 1);
        for s = 1:nStimuli % Stimulus
            coordinateMatrix = ...
                [horOnsetCoord(g), vertOnsetCoord(g); ...
                 horStimCoord(s), vertStimCoord(s)];

            distanceToStimulus(s) = pdist(coordinateMatrix, 'euclidean');
        end

        % Drop distance to the currently fixated stimulus (except for the
        % first gaze shift, which is at the fixation cross, and we do not
        % calculate the distance relative to the cross)
        if ~isnan(currentAoi(g)) & currentAoi(g) ~= flagBg & correctCurrent
            distanceToStimulus(currentAoi(g)) = NaN;
        end

        % Check whether gaze shift went to closest stimulus
        % If only one stimulus shown in a trial (probably the currently
        % fixated one), we will not be able to calculate any distances
        [~, idxClosest] = min(distanceToStimulus);
        if ~isempty(idxClosest)
            wentToClosest(g) = idxClosest == nextAoi(g);
            wentNotToClosest(g) = idxClosest ~= nextAoi(g) & ...
                                  nextAoi(g) ~= flagBg;
            distanceToClosest(g) = distanceToStimulus(idxClosest);
        end
    end

end