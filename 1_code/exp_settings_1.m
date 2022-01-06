%% DEBUGING
epar.DEBUG.aois = 0; % Show AOIs around stimuli and tolerance area around fixation


%% Timing related settings
epar.duration          = 390;  % Duration of experiment (seconds)
epar.timer_cum         = 0;    % Timer, tracking how much time passed during experiment
epar.feedback_dur      = 1.50; % Display duration feedback screen (seconds)
epar.maxDwellTime      = 0.50; % Maximum dwell time per stimulus (seconds)
epar.stimTurnOffOffset = 0.25; % Delay relative to time when eye left AOIleft, after which previously fixated stimulus is turned off (seconds)
epar.fb_sound_dur      = 0.50; % Duration feedback sound (seconds)


%% Audio settings
epar.fb_sound_freq = 1500; % Pitch feedback sound
epar.fb_sound_vol  = 0.80; % Volume feedback sound


%% Reward structure
epar.score   = 0;    % Starting score
epar.rewardE = 0.02; % Reward easy target (Cents)
epar.rewardD = 0.02; % Reward hard target


%% Visual stimulus releated settings
epar.aoiSize  = 3;    % Size of AOI around stimuli centers (deg), within which eye has to land in order to change mask to stimulus
epar.pic_size = 49;   % Stimulus size (pixel)
epar.stimCont = 0.15; % Factor, by which we adjust the contrast of the
                      % rectangular element within the stimulus
                      % 0 == invisible; 1 == fully visible

% Indices in texture matrix, which indicate locations of
% horizontal/vertical stimuli
% horizontal = gap is either up/down
% vertical   = gap is either left/right
epar.targHor_idx  = [1 4]; % Horizontal
epar.targVert_idx = [2 3]; % Vertical

% Stimulus difficulty
epar.targDiff_easy = 10;                 % Level easy target
epar.targDiff_hard = 9;                  % Level hard target
% epar.diff2         = epar.targDiff_easy; % Gap size of the easy target
% epar.diff3         = epar.targDiff_hard; % Gap size of the difficult target

% # targets per trial
if epar.expNo == 2

    epar.targ = 1;

elseif epar.expNo == 3

    epar.targ = 2;

end

%% On-screen locations of stimuli
epar.distMin = 4;            % Min. distance between two neighboring stimuli (deg)
epar.spread  = 8;            % Horizontal bounds of area, within which stimuli can appear (i.e., largest possible horizontal distance relative to fixation cross location)
epar.fixOff  = epar.distMin; % Vertical offset between fixation cross location and lower bound of area, within which stimuli can appear

% Generate grid with possible stimulus locations
% For x-positions, we just draw some position on both sides of the
% horizontal screen center. For y-positions, we only draw positions above
% the fixation cross; since here, stimuli are only distributed on one side
% of fixation, we take twice the spread to draw locations, in order to get
% a square stimulus area. Additionally, we shift the drawn y-positions by a
% little bit, to account for the minimum distance to the fixation we intend
% to implement. The latter is necessary so that the fixation check area as
% well as the AOIs of stimuli do not overlap
epar.x = (epar.spread - epar.spread * -1) .* rand(1000, 1) + epar.spread * -1;
epar.y = ((2 * epar.spread) .* rand(1000, 1)) + epar.fixOff;


%% Balancing
% Define if blue/red is easy/difficult and defines indices by which we can
% access the color of easy/difficult stimuli in arrays with mask stimuli
if mod(epar.subject, 2)         % Odd subject numbers: easy is blue, hard is red

    epar.stim.diffFlag_blueEasy = 1;

    epar.stim.idx_easyStim = 2;
    epar.stim.idx_hardStim = 1;

else                            % Even subject numbers: easy is red, hard is blue

    epar.stim.diffFlag_blueEasy = 0;

    epar.stim.idx_easyStim = 1;
    epar.stim.idx_hardStim = 2;

end


%% Generate series of shuffled miniblocks
% In Experiment 2 and 3, we present a random number of easy/hard distractors
% in each trial. Since the experiment runs a variable number of trials, we
% can't balance the levels of this factor. Therefore, we generate miniblocks
% in which we present all combinations of number of easy/hard distractors.
% In each miniblock, the order of the possible no. combinations is shuffled
noOfStimInTrial = 10;                    % Max. no. of stimuli per trial
distractorNoS   = 0:1:noOfStimInTrial-2; % Possible no. of easy distractors in a trial
if epar.expNo == 2

    % In Experiment 2 we have one target in each trial, which can either be easy
    % or hard. This target is presented with a varying number of distractors from
    % the same family
    epar.trials.disBlocks       = zeros(numel(distractorNoS)*2, 1) + epar.targ;
    epar.trials.disBlocks(:, 2) = [distractorNoS'; zeros(numel(distractorNoS), 1)];
    epar.trials.disBlocks(:, 3) = flipud(epar.trials.disBlocks(:, 2));
    epar.trials.disBlocks(:, 4) = [zeros(numel(distractorNoS), 1) + 1; ...
                                   zeros(numel(distractorNoS), 1) + 2];

elseif epar.expNo == 3

    % In Experiment 3 we show both targets in each trial. Additionally, we
    % present a varying number of easy and hard distractors
    epar.trials.disBlocks       = [zeros(numel(distractorNoS), 1) + epar.targ ...
                                   distractorNoS'];
    epar.trials.disBlocks(:, 3) = noOfStimInTrial - ...
                                  epar.trials.disBlocks(:, 1) - ...
                                  epar.trials.disBlocks(:, 2);
    epar.trials.disBlocks(:, 4) = NaN;

end

% Generate an arbitrary number of shuffled miniblocks
no_miniblocks = 1000;
no_disLvl     = numel(distractorNoS);

epar.trials.disBlocksRand = [];
for p = 1:no_miniblocks % Permutation

    % Add shuffled miniblock
    perm_disLvl = randperm(size(epar.trials.disBlocks, 1))';

    epar.trials.disBlocksRand = [epar.trials.disBlocksRand; ...
                                 epar.trials.disBlocks(perm_disLvl, :)];

end


%% Misc. settings
no_trials = size(epar.trials.disBlocksRand, 1);

epar.diff              = epar.trials.disBlocksRand(:, 4);              % Target in trial
epar.trials.targ       = zeros(no_trials, 1) + epar.targ;              % # of targets per trial
epar.trials.stairSteps = [zeros(no_trials, 1) + epar.targDiff_easy ...   Target difficulty
                          zeros(no_trials, 1) + epar.targDiff_hard];