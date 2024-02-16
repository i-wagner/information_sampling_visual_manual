function [isMissing, subjectId] = getSubjectId(subjectNumber, conditionNumber, pathData)

    % Generates identifier for subject.
    % Can be used to localise data folder as well as log-file.
    %
    % Input
    % subjectNumber: integer; ID of this subject
    % conditionNumber: integer; ID of condition
    % pathData: string; data to folder with data
    %
    % Output
    % isMissing: boolean; is subject witht his ID missing?
    % subjectId: string; ID of this subject. Is empty if subject is
    % missing

    %% Get subject ID
    subjectId = sprintf('e%dv%db%d', conditionNumber, subjectNumber);

    isMissing = false;
    dataFolders = dir(strcat(pathData, subjectId, '*'));
    if ~isempty(dataFolders)
        % Sometimes subjects have multiple data folders, for example, if
        % they did multiple blocks, because the experiment crashed during
        % one block. In this case, just pick the folder with the highest 
        % block number, assuming this is the most recent data
        % set.
        subjectId = dataFolders(end).name;
    else
        isMissing = true;
    end

end