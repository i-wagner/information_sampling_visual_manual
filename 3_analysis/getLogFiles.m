function logs = getLogFiles(exper, logCol)

    % Wrapper function
    % Performs various steps to load and process log files of participants
    % in conditions
    %
    % The following steps are performed:
    % - Load log file
    % - Recode fixation error column in log file of conditions in manual
    %   search experiment from NaN to Boolean
    % - Re-adjust vertical stimulus coordinates so they are centered on the
    %   fixation location (only manual search experiment)
    % - Recode number of distractors in single-target condition so it is
    %   unambigous how many distractors from which set where shown in a
    %   trial
    % - Recode gap position in the single-target condition, so it is
    %   unambigous which target was shown in a trial
    % - Determine how many trials a participant completed
    % - Save some relevant columns from the log file in their own variable
    %   (for easier processing later on)
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % logCol:
    % structure; column indices for log files, as returned by the
    % "settings_log" script
    % 
    % Output
    % logs:
    % structure; log-files of participants in conditions as well number of
    % completed trials and variables with data from relevant columns

    %% Get log files
    logs.files = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    logs.nCompletedTrials = NaN(exper.n.SUBJECTS, exper.n.CONDITIONS);
    logs.error.fixation.online = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        thisCondition = exper.num.CONDITIONS(c);
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
    
            % Load log file
            [thisSubject.logFile, thisSubject.isMissing] = ...
                loadLog(thisSubject.number, thisCondition, exper.path.DATA);
            if thisSubject.isMissing
                continue
            end

            %  Recode fixation error column in conditions of manual search
            %  condition. This is done so the data is line with the visual
            %  search condition, and to simplify analysis later on
            if ismember(c, [3, 4])
                thisSubject.logFile(:,logCol.IS_FIXATION_ERROR) = false;
            end

            % Adjust vertical stimulus position
            % Degree-of-visual-angle coordinates can be expressed in different 
            % reference frames, e.g., relative to the fixation cross position
            % or relative to the screen center. In the VISUAL SEARCH
            % experiment, they are expressed relative to the fiaxtion cross,
            % while in the MANUAL SEARCH experiment, they are expressed
            % relarive to screen center. Here, we correct coordinates in the
            % manual search experiment so they are in line with how coordinates
            % are expressed in the visual search experiment
            if any(thisCondition == [4, 5]) % Manual search
                thisSubject.logFile(:,logCol.STIMULUS_POSITION_Y) = ...
                    adjustVerticalCoordinates(thisSubject.logFile(:,logCol.STIMULUS_POSITION_Y), ...
                                              exper.fixation.location.y.DVA);
            end
    
            % Recode distractor numbers in single-target condition.
            % In the single-target condition, trials in which the easy
            % target was shown without distractors are coded with 0 in the
            % column in the column, which logs the number of difficult
            % distractors in a trial. The same is true for trials where the
            % difficult target was shown. This is ambigous, because, in this
            % condition, no distractors from the respective other set where
            % shown at all! To make it easier to find trials in the
            % single-target condition, where one of the target was shown
            % without distractors, we recode the column with the number of
            % easy/difficult distractors in a trial to be NaN for the
            % distractor of the non-shown set, i.e., difficult when easy was
            % shown, and easy when difficult was shown
            if mod(thisCondition, 2) == 0
                idx = [logCol.N_DISTRACTOR_EASY, ...
                       logCol.N_DISTRACTOR_DIFFICULT];
    
                thisSubject.logFile(:,idx) = ...
                    recodeDistractorNumber(thisSubject.logFile(:,idx), ...
                                           thisSubject.logFile(:,logCol.DIFFICULTY_TARGET), ...
                                           [exper.stimulus.id.target.EASY, ...
                                            exper.stimulus.id.target.DIFFICULT]);
                clear idx
            end
    
            % Recode gap position
            % In the single-target condition, the gap position of the shown
            % target is stored in the column, which, in the double-target
            % condition, houses the gap position on the easy target. We recode
            % this and align the storage scheme in the single-target with the
            % scheme in the double-target condition, to make analysis more
            % straightforward
            thisSubject.logFile(:,logCol.GAP_POSITION_EASY:logCol.GAP_POSITION_DIFFICULT) = ...
                recodeGapPosColumns(thisSubject.logFile(:,6), ...
                                    [thisSubject.logFile(:,logCol.GAP_POSITION_EASY), ...
                                     thisSubject.logFile(:,logCol.GAP_POSITION_DIFFICULT)]);
    
            % Store data for output
            logs.files{thisSubject.number,c} = thisSubject.logFile;
            logs.nCompletedTrials(thisSubject.number,c) = ...
                max(thisSubject.logFile(:,logCol.TRIAL_NO));
            logs.error.fixation.online{thisSubject.number,c} = ...
                thisSubject.logFile(:,logCol.IS_FIXATION_ERROR);
            clear thisSubject
        end
    end
end