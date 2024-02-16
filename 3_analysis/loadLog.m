function [logFile, isMissing] = loadLog(thisSubject, thisCondition, pathData)

    % Loads log file of the current subject
    %
    % Input
    % thisSubject: integer; ID of this subject
    % thisCondition: integer; ID of this condition
    % pathData: string; path to folder with data
    %
    % Output
    % logFile: matrix; log file of subject

    %% Load log file of subject
    [isMissing, subjectId] = ...
        getSubjectId(thisSubject, thisCondition, pathData);

    logFile = [];
    if ~isMissing
        thisFilename = strcat(pathData, subjectId, '/', subjectId, '.log');
        logFile = readmatrix(thisFilename);
    end

end