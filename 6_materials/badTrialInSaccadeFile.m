clear all; close all; clc

%% Init
pathData = "/Users/ilja/Dropbox/12_work/" + ...
           "mr_informationSamplingVisualManual/2_data/";

opts = detectImportOptions(strcat(pathData, "BadTrials.csv"));
opts = setvartype(opts, ...
                 {'Var1', 'Participant', 'Trial', 'Task'}, ...
                 {'double', 'double', 'double', 'categorical'});
badTrials = readtable(strcat(pathData, "BadTrials.csv"), opts);
nEntries = size(badTrials, 1);

%% Check whether bad trials are included in saccades file
% In other words: checks whether a bad trial was already flagged in Jan'S
% saccade file
badTrialIncluded = false(nEntries, 1);
for e = 1:nEntries % Entries
    thisSubjectNumber = table2array(badTrials(e,2));
    thisTrialNumber = table2array(badTrials(e,3));
    thisCondition = string(table2array(badTrials(e,4)));

    if strcmp(thisCondition, "single")
        condId = num2str(4);
    elseif strcmp(thisCondition, "double")
        condId = num2str(5);
    end
    subjectId = strcat("e", condId, ...
                       "v", num2str(thisSubjectNumber), ...
                       "b1");
    gazeShifts = readmatrix(strcat(pathData, subjectId, "/", subjectId, "_saccades.csv"));
    badTrialIncluded(e) = any(gazeShifts(:,17) == thisTrialNumber);
end