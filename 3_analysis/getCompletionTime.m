function conditionDurations = getCompletionTime(exper, anal, filename)

    % Loads timestamps from "filename" and calculates time to complete
    % conditions.
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
    % filename:
    % string; path to file with time timestamps. "filename" has to be a 
    % .mat file, as created by the "getTimestamps" function
    %
    % Output
    % conditionDurations:
    % matrix; time (in minutes), participants took to complete conditions.
    % Rows are participant numbers, columns are condition

    %% Get condition durations
    timestamps = struct2array(load(filename));
    conditionDurations = minutes(NaN(exper.n.SUBJECTS, size(timestamps, 3)));
    for c = 1:size(timestamps, 3) % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.idx = vertcat(timestamps{:,1,c,1}) == thisSubject.number;
            if ismember(thisSubject.number, anal.excludedSubjects) | ... 
                all(~thisSubject.idx)
                continue;
            end

            % "timestamps" will be a 4D array:
            % -- subjects
            % -- [subject number, condition id, timestamps]
            % -- condition
            % -- block (1: main, 2: demo)
            thisSubject.timestamps = timestamps{thisSubject.idx,3,c,1};
    
            % Duration is simple the time-difference between the "last
            % modified" timestamp of the first and last trial of a condition
            conditionDurations(thisSubject.number,c) = ...
                thisSubject.timestamps(end) - thisSubject.timestamps(1);
        end
    end
end