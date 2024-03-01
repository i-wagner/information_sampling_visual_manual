%% General analysis
anal.average.MIN_N = 10;

%% Saccade detection
anal.saccadeDetection.MIN_SACC_DUR = 5;

%% Fixation check
anal.fixation.tolerance.DVA = 1.50;
anal.fixation.checkBounds = [-20, 80];

%% Calculation of dwell times
% Depending on the expeirmental condition, we either do or do not consider
% fixations on targets, when calculating dwell times. #
% We have four conditions: single-target visual, double-target visual,
% single-target manual, double-target manual
anal.dwellTimes.useTargets = [false, true, false, true];

%% Event detection
% We expect five events to happen in each trial of both, the visual and
% manual search experiment:
% - trial begin
% - start recording
% - fixation cross onset
% - onset of stimuli
% - offset of stimuli (i.e., response)
% 
% For manual search the first two events are coded as NaN, because they did
% not exist in the experiment. Additionally, the "fixation cross onset"
% event is coded as zero, and serves as a reference point, relative to
% which all other event timestamps are calculated
%
% In manual search, we have NO sample numbers, instead, we only have access
% to the timestamps of events
anal.nExpectedEvents = 5;