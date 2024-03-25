function fixations = getFixatedAois(exper, screen, anal, gaze, stimCoords, nTrials, showFix)

    % Wrapper function
    % Extracts fixated areas of interest.
    % The following analysis steps are performed:
    % - Get fixated AOIs
    % - Selects a subset of gaze shifts to use for subsequent analysis.
    %   This selection is, for one, based on quality criteria (see below),
    %   and on some task-related selection criteria
    % - Check for blinks during AOI vists, and calculate information loss
    %   during blinks
    % - Check whether at least one gaze shift was made to any AOI
    % - Check whether gaze shifts went to closest AOI, relative to the
    %   current fixation location
    % - Check the distance between gaze and the currently viewed AOI
    %
    % NOTE 1:
    % Selection of gaze shift subset happens here, instead of some seperate
    % function, because of two reasons: first, one of the criteria (see
    % below) for including gaze shifts in the subset is the fixated AOI,
    % which also determined within this wrapper function. Second, some
    % fixation-related  analysis, performed within this wrapper function,
    % need information about the selected subset
    %
    % NOTE 2:
    % The following quality criteria are applied to select gaze shift
    % subset:
    % - Gaze shift is longer than some minimum duration
    % - Gaze shift has both, on- and offset
    % - Gaze shift offset occurs before a response is placed (i.e., before
    %   the end of a trial)
    % - Gaze shift on- and offset coordinate are within the measurable
    %   screen area
    %
    % NOTE 3:
    % Additionally, the following task-related criteria are applied when
    % selecting a subset of gaze shifts
    % - Gaze shift onset occured after stimulus onset
    % - Consecutive gaze shifts within an AOI are excluded (i.e., gaze
    %   shifts that occured within an AOI)
    % - If the last gaze shift in a trial landed on the background (i.e.,
    %   outside any AOI), this gaze shift is excluded
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % screen:
    % structure; settings of screen, on which experiment was recorded, as 
    % returned by the "settings_screen" script
    %
    % anal:
    % structure; various analysis settings, as returned by the
    % "settings_analysis" script
    %
    % gaze:
    % structure; gaze data of participants in conditions
    %
    % stimCoords:
    % cell-matrix; coordinates of stimuli, as returned by the
    % "getStimCoord" function
    %
    % nTrials:
    % matrix; number of completed trials per participant and condition
    %
    % showFix:
    % Boolean; visualise shown stimuli and fixated AOI?
    %
    % Ouput
    % fixations:
    % structure; fixated AOIs across participants and conditions

    %% Analyse fixations
    fixations.fixatedAois.groupIds = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    fixations.fixatedAois.uniqueIds = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    fixations.subset = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    fixations.informationLoss = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    fixations.atLeastOneFixatedAoi = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    fixations.wentToClosest = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    fixations.propToClosest = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    fixations.distanceCurrent = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
            if isnan(thisSubject.nTrials)
                continue
            end

            thisSubject.fixatedAois.groupIds = [];
            thisSubject.fixatedAois.uniqueIds = [];
            thisSubject.fixationSubset = [];
            thisSubject.informationLoss = [];
            thisSubject.atLeastOneFixatedAoi = NaN(thisSubject.nTrials, 1);
            thisSubject.wentToClosest = [];
            thisSubject.propToClosest = NaN(thisSubject.nTrials, 1);
            thisSubject.distanceCurrent = [];
            for t = 1:thisSubject.nTrials % Trial
                % Unpack trial data
                thisTrial.idx = ...
                    gaze.gazeShifts.trialMap{thisSubject.number,c} == t;
                thisTrial.gazeShifts.idx = ...
                    gaze.gazeShifts.idx{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.gazeShifts.meanGazePos = ...
                    gaze.gazeShifts.meanGazePos{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.stimulusCoordinates = ...
                    squeeze(stimCoords{thisSubject.number,c}(t,:,:));
                thisTrial.gazeShifts.onsets = ...
                    gaze.gazeShifts.onsets{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.gazeShifts.offsets = ...
                    gaze.gazeShifts.offsets{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.gazeShifts.duration = ...
                    gaze.gazeShifts.duration{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.timestamp.stimOn = ...
                    gaze.timestamps.stimOn{thisSubject.number,c}(t,:);
                thisTrial.timestamp.stimOff = ...
                    gaze.timestamps.stimOff{thisSubject.number,c}(t,:);

                % Get fixated AOIs
                % We are using the mean gaze position inbetween gaze shifts
                % for that. We do this, because, sometimes, a gaze shift
                % might initially land in an AOI/land close to the edge of 
                % an AOI, but then drift out/drift into the AOI. By using 
                % the mean gaze position after each gaze shift we can 
                % circumvent those kind of fluctuations and get more 
                % reliable estimate for if an AOI was fixated or not
                [thisTrial.fixatedAois.uniqueIds, thisTrial.fixatedAois.groupIds] = ...
                    getFixatedAOI(thisTrial.gazeShifts.meanGazePos(:,1), ...
                                  thisTrial.gazeShifts.meanGazePos(:,3), ...
                                  thisTrial.stimulusCoordinates(:,1), ...
                                  thisTrial.stimulusCoordinates(:,2), ...
                                  exper.stimulus.aoi.radius.DVA, ...
                                  exper.stimulus.id.BACKGROUND, ...
                                  exper.stimulus.id.distractor.EASY, ...
                                  exper.stimulus.id.distractor.DIFFICULT);
                if showFix
                    plotStimulusPositions(thisTrial.gazeShifts.meanGazePos(:,1), ...
                                          thisTrial.gazeShifts.meanGazePos(:,3), ...
                                          thisSubject.stimulusCoordinates(t,:,1), ...
                                          thisSubject.stimulusCoordinates(t,:,2))
                end
    
                % Select fixations for analysis
                [thisTrial.fixationSubset, thisTrial.qualityPassed] = ...
                    selectFixationSubset(thisTrial.fixatedAois.uniqueIds, ...
                                         thisTrial.gazeShifts.onsets(:,1), ...
                                         thisTrial.gazeShifts.offsets(:,1), ...
                                         [thisTrial.gazeShifts.onsets(:,2), ...
                                          thisTrial.gazeShifts.offsets(:,2), ...
                                          thisTrial.gazeShifts.meanGazePos(:,1)], ...
                                         [thisTrial.gazeShifts.onsets(:,3), ...
                                          thisTrial.gazeShifts.offsets(:,3), ...
                                          thisTrial.gazeShifts.meanGazePos(:,3)], ...
                                         thisTrial.gazeShifts.duration, ...
                                         thisTrial.timestamp.stimOn, ...
                                         thisTrial.timestamp.stimOff, ...
                                         exper.stimulus.id.BACKGROUND, ...
                                         screen.bounds.dva, ...
                                         anal.saccadeDetection.MIN_SACC_DUR);

                % Check for blinks during AOI visits
                % If a participant blinked during an AOI visit, and the 
                % gaze was in the same AOI before and after the blink, we 
                % will adjust the dwell time by the duration of the blink 
                % (since participants cannot see anything during the blink, 
                % and thus, technially do not "dwell" during this time)
                thisTrial.informationLoss = ...
                    getInformationLoss(thisTrial.fixatedAois.uniqueIds, ...
                                       thisTrial.fixationSubset, ...
                                       logical(thisTrial.gazeShifts.idx(:,3)), ...
                                       thisTrial.gazeShifts.duration, ...
                                       thisTrial.qualityPassed);
    
                % Check whether at least one gaze shift was made to any AOI
                thisTrial.atLeastOneFixatedAoi = ...
                    checkOneAoiGazeShift(thisTrial.fixatedAois.uniqueIds(thisTrial.fixationSubset), ...
                                         exper.stimulus.id.BACKGROUND);
    
                % Check if gaze shifts went to closest stimulus.
                % We are using gaze shift onset as reference here, because we
                % want to know which stimulus was closest immediately before
                % people made the gaze shift. We are currently only calculating
                % the mean gaze position between two gaze shifts, so we cannot
                % use this as the reference (since it describes the position
                % AFTER the gaze shift was already made)
                thisTrial.wentToClosest = ...
                    getDistanceToClosestStim(thisTrial.fixatedAois.uniqueIds, ...
                                             thisTrial.stimulusCoordinates(:,1), ...
                                             thisTrial.stimulusCoordinates(:,2), ...
                                             thisTrial.gazeShifts.onsets(:,2), ...
                                             thisTrial.gazeShifts.onsets(:,3), ...
                                             exper.stimulus.id.BACKGROUND, ...
                                             true);
                thisTrial.propToClosest = ...
                    mean(thisTrial.wentToClosest(thisTrial.fixationSubset), 'omitnan');

                % Check distance between gaze and the currently fixated stimulus.
                % We are using the mean gaze position between gaze shifts as
                % reference, because we want to know how closely gaze stayed to
                % a stimulus, while fixating it
                [~, ~, thisTrial.distanceCurrent] = ...
                    getDistanceToClosestStim(thisTrial.fixatedAois.uniqueIds, ...
                                             thisTrial.stimulusCoordinates(:,1), ...
                                             thisTrial.stimulusCoordinates(:,2), ...
                                             thisTrial.gazeShifts.meanGazePos(:,1), ...
                                             thisTrial.gazeShifts.meanGazePos(:,3), ...
                                             exper.stimulus.id.BACKGROUND, ...
                                             false);

                % Store data
                thisSubject.fixatedAois.groupIds = ...
                    [thisSubject.fixatedAois.groupIds; thisTrial.fixatedAois.groupIds];
                thisSubject.fixatedAois.uniqueIds = ...
                    [thisSubject.fixatedAois.uniqueIds; thisTrial.fixatedAois.uniqueIds];
                thisSubject.fixationSubset = ...
                    [thisSubject.fixationSubset; thisTrial.fixationSubset];
                thisSubject.informationLoss = ...
                    [thisSubject.informationLoss; thisTrial.informationLoss];
                thisSubject.atLeastOneFixatedAoi(t) = thisTrial.atLeastOneFixatedAoi;
                thisSubject.wentToClosest = ...
                    [thisSubject.wentToClosest; thisTrial.wentToClosest];
                thisSubject.propToClosest(t) = thisTrial.propToClosest;
                thisSubject.distanceCurrent = ... 
                    [thisSubject.distanceCurrent; thisTrial.distanceCurrent];
                clear thisTrial
            end

            % Store data
            fixations.fixatedAois.groupIds{thisSubject.number,c} = ...
                thisSubject.fixatedAois.groupIds;
            fixations.fixatedAois.uniqueIds{thisSubject.number,c} = ...
                thisSubject.fixatedAois.uniqueIds;
            fixations.subset{thisSubject.number,c} = ...
                thisSubject.fixationSubset;
            fixations.informationLoss{thisSubject.number,c} = ...
                thisSubject.informationLoss;
            fixations.atLeastOneFixatedAoi{thisSubject.number,c} = ...
                thisSubject.atLeastOneFixatedAoi;
            fixations.wentToClosest{thisSubject.number,c} = ...
                thisSubject.wentToClosest;
            fixations.propToClosest{thisSubject.number,c} = ...
                thisSubject.propToClosest;
            fixations.distanceCurrent{thisSubject.number,c} = ...
                thisSubject.distanceCurrent;
            clear thisSubject
        end
    end

end