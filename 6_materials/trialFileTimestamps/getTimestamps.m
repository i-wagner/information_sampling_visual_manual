% Extracts timestamps of when files were last modified.
% "lastModifiedDate" is a 4D cell array with the following columns
% - Subject ID (as used to code data folder)
% - Condition ID (as used to code data folder)
% - Timetamps
% 
% Rows are subjects, third array dimension are different conditions, fourth
% array dimension are different blocks
clear all; close all; clc

%% Init
% pathToData is path to folder, where all data folder with the format 
% "e*v*b*" lie. pathToData needs to house data from both conditions of one 
% experiment 
thisExp = "visual";
blocks = [1, 99]; % 99 is demmo trials, 1 is main measurement
nBlocks = numel(blocks);

pathToData = "/Users/ilja/Desktop/dataFromLab_eyeTracking/";
cd(pathToData);

%% Get timestamps
for b = 1:nBlocks % Block
    folder = dir(strcat(pathToData, 'e*b', num2str(blocks(b))));
    folder = natsortfiles(folder); % Natural sort
    nFolder = size(folder, 1);

    idxCond = 1;
    idxFolder = 1;
    for f = 1:nFolder % Folder
        thisFolder.name = folder(f).name;
        thisFolder.subjectNumber = ...
            str2double(extractBetween(thisFolder.name, ...
                                      "e" + digitsPattern(1) + "v", ...
                                      "b" + digitsPattern(1)));
        thisFolder.conditionNumber = ...
            str2double(extractBetween(thisFolder.name, "e" , "v"));
        thisFolder.files = natsortfiles(dir(strcat(thisFolder.name, '/*.dat')));
        thisFolder.nTrials = numel(thisFolder.files);
        
        thisFolder.lastModifiedDate = NaT(thisFolder.nTrials, 1);
        for t = 1:thisFolder.nTrials % Trial
            % Call Python functions to extract the last modified date of a file
            thisFolder.filepath = strcat(thisFolder.name, "/", thisFolder.files(t).name);
            thisFolder.lastModifiedDate(t) = ...
                datetime(py.os.path.getmtime(thisFolder.filepath), ...
                         'ConvertFrom', 'epochtime', ...
                         'TicksPerSecond', 1, ...
                         'Format', 'dd-MMM-yyyy HH:mm:ss.SSS');
        end
        lastModifiedDate{idxFolder,1,idxCond,b} = thisFolder.subjectNumber;
        lastModifiedDate{idxFolder,2,idxCond,b} = thisFolder.conditionNumber;
        lastModifiedDate{idxFolder,3,idxCond,b} = thisFolder.lastModifiedDate;
    
        if (f < nFolder)
            idxFolder = idxFolder + 1;
            if str2double(folder(f).name(2)) - ...
               str2double(folder(f+1).name(2)) ~= 0
                % Reset row-index and increment column-index for 
                % double-target condition
                idxFolder = 1;
                idxCond = idxCond + 1;
            end
        end
        clear thisFolder
    end
end

%% Save timestamps
save(strcat("timestamps_", thisExp, ".mat"), "lastModifiedDate");
