%% Data of which subject from which condition to export?
conditionOfInterest = 2; % Double-target
subjectOfInterest = 7;

%% Get gaze shifts
colsOfInterest = [4, 7, 13, 15, 17, 18, 22:24, 26];
gazeShiftsForAlexander = sacc.gazeShifts{subjectOfInterest,conditionOfInterest};
gazeShiftsForAlexander = gazeShiftsForAlexander(:,colsOfInterest);

%% Get stimulus locations
colsOfInterest = 30:66;
stimPosForAlexander = sacc.gazeShifts_zen{subjectOfInterest,conditionOfInterest};
stimPosForAlexander = stimPosForAlexander(:,colsOfInterest);
[~, idxTrials] = unique(stimPosForAlexander(:,1));
nTrials = numel(idxTrials);
stimPosForAlexander = stimPosForAlexander(idxTrials,2:end);

colsHorizontalPositions = 1:18;
colsVerticalPositions = 19:36;
stimPosForAlexander = cat(3, ...
                          stimPosForAlexander(:,colsHorizontalPositions), ...
                          stimPosForAlexander(:,colsVerticalPositions));

%% Response correctness
% FOR RESPONSES WE WILL USE DATA FROM SINGLE-TARGET
conditionOfInterest = 1;

thisResponseData = perf.hitMiss{subjectOfInterest,conditionOfInterest};
thisChosenTarget = stim.chosenTarget{subjectOfInterest,conditionOfInterest};

responseChosenTarget = [thisChosenTarget, thisResponseData];

%% Validate gaze data and stimulus locations
aoiRadius = stim.radiusAOI.deg;
debugPlot = [0, 0];

fixatedAois = [];
for t = 1:nTrials
    liThoseGazeShifts = gazeShiftsForAlexander(:,10) == t;
    thoseHorizontalEndpoints = gazeShiftsForAlexander(liThoseGazeShifts,3);
    thoseVerticalEndpoints = gazeShiftsForAlexander(liThoseGazeShifts,4);
    thoseHorizontalLocations = stimPosForAlexander(t,:,1);
    thoseVerticalLocations = stimPosForAlexander(t,:,2);

    thoseAois = infSampling_getFixatedAOI(thoseHorizontalEndpoints, thoseVerticalEndpoints, ...
                                          thoseHorizontalLocations, thoseVerticalLocations, ...
                                          aoiRadius, stim.identifier_bg, ...
                                          debugPlot);
    fixatedAois = [fixatedAois; thoseAois];
end

resultReplicated = all((gazeShiftsForAlexander(:,5) == fixatedAois));


%% Export
exportPath = ['/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual/' ...
              '6_materials/dataExportforAlexander/'];
errorMsg = 'Cannot export data, because to-be-exported data cannot replicate analysis!';
if resultReplicated
    save([exportPath, 'gazeShiftsForAlexander.mat'], 'gazeShiftsForAlexander');
    save([exportPath, 'stimPosForAlexander.mat'], 'stimPosForAlexander');
    save([exportPath, 'responseCorrectnessForAlexander.mat'], 'responseChosenTarget')
else
    error(errorMsg);
end

%% Load exported data
importedGazeShifts = load([exportPath, 'gazeShiftsForAlexander.mat']);
importedStimPos = load([exportPath, 'stimPosForAlexander.mat']);
importedResponseData = load([exportPath, 'responseCorrectnessForAlexander.mat']);