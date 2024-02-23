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