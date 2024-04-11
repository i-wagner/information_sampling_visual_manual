function excludedTrials = getExcludeTrials(exper, anal, isOnlineFixErr, isOfflineFixErr, isDataLoss, isMissingEvent, isPenDragging)

    % Determine which trials to exclude from subsequent analysis
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % anal:
    % structure; vairous analysis settings, as returned by the
    % "settings_analysis" script
    % 
    % isOnlineFixErr:
    % matrix; Booleans indicating which trial for which subject in which
    % condition contained an fixation error that was detected online
    % 
    % isOfflineFixErr:
    % matrix; Booleans indicating which trial for which subject in which
    % condition contained an fixation error that was detected offline
    % 
    % isDataLoss:
    % matrix; Booleans indicating trials where dataloss occured for each
    % subject and each condition
    % 
    % isMissingEvent:
    % matrix; Booleans indicating trials where Eye Link events were missing
    % for each subject and each condition
    % 
    % isPenDragging:
    % matrix; Booleans indicating trials where pen dragging occured (only
    % defined for manual search conditions)
    %
    % Output
    % excludedTrials:
    % matrix; numbers of trials which are flagged for exclusion

    %% Check which trials should be excluded
    excludedTrials = cell(exper.n.SUBJECTS, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            if ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end

            idx.fixErr.online = find(isOnlineFixErr{thisSubject.number,c});
            idx.fixErr.offline = find(isOfflineFixErr{thisSubject.number,c});
            idx.dataLoss = find(isDataLoss{thisSubject.number,c});
            idx.eventMissing = find(isMissingEvent{thisSubject.number,c});
            idx.penDragging = find(isPenDragging{thisSubject.number,c});
            idx.exclude = ...
                unique([idx.fixErr.online; idx.fixErr.offline; ...
                        idx.dataLoss; idx.eventMissing; idx.penDragging]);

            excludedTrials{thisSubject.number,c} = idx.exclude;
        end
    end
end
