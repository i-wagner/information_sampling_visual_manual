function time = getTimes(exper, anal, nTrials, gaze, fixations, excludedTrials, dropOutliers)

    % Wrapper function
    % Extracts trialwise planning, dwell, inspection, response, and
    % non-search time for each subject in conditions
    %
    % NOTE 1:
    % This wrapper function uses the SUBSET of fixation, as determined in
    % the "getFixatedAois" function, to calculate all output variables
    %
    % NOTE 2:
    % Dwell times are saved in a vector with it's length corresponding to
    % the overall number of fixations that were made in the trial (i.e.,
    % not only the subset). This is done to later simplify indexing into
    % the dwell-time array (i.e., we don't need a specific trial map to do
    % this). We chose this approach instead of calculating the dwell time
    % for all fixations that were made, because dwell-times are calculated
    % using the leaving times of AOIs. Leaving times, however, differ,
    % depending on whether all or only a subset of fixations is used to
    % calculate them (since leaving times are calculated based on the
    % specific sequence of inspected AOIs, which naturally diffres
    % depending on which fixations are used to calculate them)
    %
    % NOTE 3:
    % non-search time is defined as the trialwise sum of the respective
    % response and planning time. It is used to quantify the overall time
    % in a trial that participants did not spend searching for a target
    % (i.e., fixating stimuli)
    %
    % NOTE 4:
    % proportion of trials where the response is missing are calculated
    % BEFORE dropping outlier trials
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % anal:
    % structure; various analysis settings, as returned by the
    % "settings_analysis" script
    %
    % nTrials:
    % matrix; number of completed trials per participant and condition
    %
    % gaze:
    % structure; gaze data of participants in conditions
    %
    % fixations:
    % structure; fixated AOIs across participants and conditions
    %
    % dropOutliers:
    % Boolean; drop outliers from trialwise data or not
    %
    % Output
    % time:
    % structure; time-variables across participants and conditions

    %% Analyse fixations
    time.inspection.trialwise = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    time.dwell.trialwise = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    time.planning.trialwise = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    time.response.trialwise = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    time.nonSearch.trialwise = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    time.propTrialsWithResp = NaN(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
            thisSubject.excludedTrials = excludedTrials{thisSubject.number,c};
            thisSubject.nGazeShifts = numel(gaze.gazeShifts.trialMap{thisSubject.number,c});
            if ismember(thisSubject.number, anal.excludedSubjects) | ...
               isnan(thisSubject.nTrials)
                continue
            end

            thisSubject.inspectionTime = NaN(thisSubject.nTrials, 1);
            thisSubject.dwellTimes = NaN(thisSubject.nGazeShifts,1);
            thisSubject.planningTime = NaN(thisSubject.nTrials, 1);
            thisSubject.responseTime = NaN(thisSubject.nTrials, 1);
            thisSubject.gazeShiftCounter = 0;
            for t = 1:thisSubject.nTrials % Trial
                % Check whether to skip excluded trial
                if ismember(t, thisSubject.excludedTrials)
                    continue
                end

                % Unpack trial data
                thisTrial.idx = ...
                    gaze.gazeShifts.trialMap{thisSubject.number,c} == t;
                thisTrial.gazeShifts.idx = ...
                    gaze.gazeShifts.idx{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.gazeShifts.onsets = ...
                    gaze.gazeShifts.onsets{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.gazeShifts.offsets = ...
                    gaze.gazeShifts.offsets{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.timestamp.stimOn = ...
                    gaze.timestamps.stimOn{thisSubject.number,c}(t,:);
                thisTrial.timestamp.stimOff = ...
                    gaze.timestamps.stimOff{thisSubject.number,c}(t,:);
                thisTrial.fixations.subset = ...
                    logical(fixations.subset{thisSubject.number,c}(thisTrial.idx,:));
                thisTrial.fixations.groupIds = ...
                    fixations.fixatedAois.groupIds{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.fixations.uniqueIds = ...
                    fixations.fixatedAois.uniqueIds{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.gazeShifts.informationLoss = ...
                    fixations.informationLoss{thisSubject.number,c}(thisTrial.idx,:);
                thisTrial.nGazeShifts = size(thisTrial.gazeShifts.idx, 1);

                % Get leaving times of AOIs
                thisTrial.leavingTimes = ...
                    getLeavingTimes(logical(thisTrial.gazeShifts.idx(thisTrial.fixations.subset,3)), ...
                                    thisTrial.gazeShifts.offsets(thisTrial.fixations.subset,1), ...
                                    thisTrial.gazeShifts.onsets(thisTrial.fixations.subset,1), ...
                                    thisTrial.timestamp.stimOff);

                % Get inspection and dwell times
                [thisTrial.inspectionTime, temp] = ...
                    getInspectionTime(thisTrial.fixations.groupIds(thisTrial.fixations.subset), ...
                                      [exper.stimulus.id.target.EASY, exper.stimulus.id.target.DIFFICULT], ...
                                      exper.stimulus.id.BACKGROUND, ...
                                      thisTrial.gazeShifts.offsets(thisTrial.fixations.subset,1), ...
                                      thisTrial.gazeShifts.informationLoss(thisTrial.fixations.subset), ...
                                      thisTrial.leavingTimes, ...
                                      anal.dwellTimes.useTargets(c));
                thisTrial.dwellTimes = NaN(size(thisTrial.fixations.groupIds));
                thisTrial.dwellTimes(thisTrial.fixations.subset) = temp;

                % Get planning time
                thisTrial.planningTime = ...
                    getPlanningTime(thisTrial.gazeShifts.offsets(thisTrial.fixations.subset,1), ...
                                    thisTrial.timestamp.stimOn);
                
                % Get response time
                thisTrial.responseTime = ...
                    getResponseTime(thisTrial.fixations.uniqueIds(thisTrial.fixations.subset), ...
                                    thisTrial.gazeShifts.offsets(thisTrial.fixations.subset,1), ...
                                    thisTrial.timestamp.stimOff, ...
                                    [exper.stimulus.id.target.EASY, exper.stimulus.id.target.DIFFICULT], ...
                                    exper.stimulus.id.BACKGROUND);

                % Store data
                thisTrial.storeIdx = ...
                    (thisSubject.gazeShiftCounter + 1):(thisSubject.gazeShiftCounter + thisTrial.nGazeShifts);
                thisSubject.gazeShiftCounter = ...
                    thisSubject.gazeShiftCounter + thisTrial.nGazeShifts;

                thisSubject.inspectionTime(t) = thisTrial.inspectionTime;
                thisSubject.dwellTimes(thisTrial.storeIdx) = thisTrial.dwellTimes;
                thisSubject.planningTime(t) = thisTrial.planningTime;
                thisSubject.responseTime(t) = thisTrial.responseTime;
                clear thisTrial
            end

            % Get number of trials without response times
            % Response time is either missing due to the trial being
            % excluded or because the last gaze shift landed on anything
            % but a target, so no response time could be calculated
            thisSubject.nTrialsNoRespTime = sum(isnan(thisSubject.responseTime));

            % Drop outliers
            if dropOutliers
                idx = isoutlier(thisSubject.planningTime);
                thisSubject.planningTime(idx) = NaN;

                idx = isoutlier(thisSubject.inspectionTime);
                thisSubject.inspectionTime(idx) = NaN;

                idx = isoutlier(thisSubject.responseTime);
                thisSubject.responseTime(idx) = NaN;
                clear idx
            end

            % Store data
            time.inspection.trialwise{thisSubject.number,c} = thisSubject.inspectionTime;
            time.dwell.trialwise{thisSubject.number,c} = thisSubject.dwellTimes;
            time.planning.trialwise{thisSubject.number,c} = thisSubject.planningTime;
            time.response.trialwise{thisSubject.number,c} = thisSubject.responseTime;
            time.nonSearch.trialwise{thisSubject.number,c} = ...
                thisSubject.planningTime + thisSubject.responseTime;
            time.propTrialsWithResp(thisSubject.number,c) = ...
                1 - (thisSubject.nTrialsNoRespTime / thisSubject.nTrials);
            clear thisSubject
        end
    end
end
