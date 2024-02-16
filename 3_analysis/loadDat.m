function [gazeTrace, hasDataloss] = loadDat(thisSubject, thisCondition, thisTrial, pathData)

    % Loads a subject's gaze trace from the corresponding dat file.
    %
    % Input
    % thisSubject: integer; ID of this subject
    % thisCondition: integer; ID of this condition
    % thisTrial: integer; number of current trial
    % pathData: string; path to folder with data
    %
    % Output
    % gazeTrace: matrix; timestamps of gaze samples, x and y gaze 
    % coordinates, and event flags.
    % hasDataloss: boolean; gaze trace is missing samples?
    %
    % NOTE
    % the exact format of gaze traces might differ, depending on which
    % playback routine was used to store them, and wheter one eye or both
    % eyes were recorded. THIS FUNCTION DOES NOT ACCOUNT FOR DIFFERENT FILE
    % FORMATS!

    %% Load dat file with gaze trace
    [~, subjectId] = ...
        getSubjectId(thisSubject, thisCondition, pathData);
    filepath = ...
        strcat(pathData, subjectId, "/trial", num2str(thisTrial), ".dat");

    gazeTrace = NaN(1, 9);
    try
        gazeTrace = load(filepath);
    catch exception
        switch exception.identifier
            case 'MATLAB:load:couldNotReadFile'
                warning(strcat("File ", ...
                                strcat(subjectId, "/trial", ...
                                       num2str(thisTrial), ".dat"), ...
                                " not found."));
            otherwise
                rethrow(exception);
        end
    end

    %% Check file integrity
    % Only applies to data that was collected in Schütz lab.
    % Sometimes gaze traces are incomplete, i.e., some random samples are
    % missing from traces. If this happens, time-stamps are non-continous
    % (assuming a 1000 Herz sampling rate, which is the default). This a
    % know bug which no one could resolve, thus, we flag the corresponding
    % trials, so we can later exclude them if we want to
    hasDataloss = any(diff(gazeTrace(:,1)) > 1);

end