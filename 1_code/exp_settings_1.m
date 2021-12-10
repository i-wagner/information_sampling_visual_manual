%% General settings
% Experiment
epar.trialNo   = 10000; % Trial number; the experiment runs a certain time,
                        % thus, we choose a high trial number to keep it
                        % running until the time runs out
epar.duration  = 390;   % Duration of experiment (in seconds)
epar.timer_cum = 0;     % Timer, tracking the overall passed time

% Timing
epar.feedback_dur      = 1.50; % Display duration of feedback screen (in seconds)
epar.maxDwellTime      = 0.50; % Maximum dwell time per stimulus (in seconds)
epar.stimTurnOffOffset = 0.10; % Delay relative to time when an AOI was left, after which a stimulus is turned off

% Rewards
epar.score   = 0;    % Starting score
epar.rewardE = 0.02; % Reward easy target (in Cents)
epar.rewardD = 0.02; % Reward hard target


%% Balancing
% Participant numbers, for which the blue stimulus is the easy one
epar.sub_blueE = 1:2:100;

% Get indices for easy/difficult mask stimuli
% Odd subject numbers: easy is blue, hard is red
if ismember(epar.subject, epar.sub_blueE)

    epar.stim.idx_easyStim = 2;
    epar.stim.idx_hardStim = 1;

% Even subject numbers: easy is red, hard is blue
else

    epar.stim.idx_easyStim = 1;
    epar.stim.idx_hardStim = 2;

end


%% Visual stimuli (targets/distractors)
epar.pic_size = 49;   % Stimulus size (in pixel)
epar.stimCont = 0.15; % Factor, by which we adjust the contrast of the
                      % rectangular element within the stimulus

% Indices for the rows in which we store the locations of gap within the
% rectangular part of the stimulus
% horizontal = gap is either up/down
% vertical = gap is either left/right
epar.targHor_idx  = [1 4]; % Horizontal
epar.targVert_idx = [2 3]; % Vertical

% Stimulus difficulty
epar.targDiff_easy = 10; % Easy target
epar.targDiff_hard = 9;  % Hard target
epar.diff2         = epar.targDiff_easy; % Gap size of the easy target
epar.diff3         = epar.targDiff_hard; % Gap size of the difficult target

% No. of targets per trial
if epar.expNo == 2

    epar.targ = 1;

elseif epar.expNo == 3

    epar.targ = 2;

end


%% On-screen locations of stimuli
% Criteria for distance between neighbouring stimuli
epar.distMin = 5;  % Min. distance between two stimuli (in deg)
epar.distMax = 30; % Max. distance between two stimuli
epar.spread  = 10; % Max. distance of a stimulus to the fixation cross 

% Generate grid with possible stimulus locations
epar.x = (epar.spread - epar.spread * -1) .* rand(1000, 1) + epar.spread * -1;
epar.y = (epar.spread - epar.spread * -1) .* rand(1000, 1) + epar.spread * -1;
epar.x = epar.x';
epar.y = epar.y';


%% Generate series of shuffled miniblocks
% In Experiment 2 and 3, we present a random number of easy/hard distractors
% in each trial. Since the experiment runs a variable number of trials, we
% can't balance the levels of this factor. Therefore, we generate miniblocks
% in which we present all combinations of number of easy/hard distractors.
% In each miniblock, the order of the possible no. combinations is shuffled
noOfStimInTrial = 10;                    % Max. no. of stimuli per trial
distractorNoS   = 0:1:noOfStimInTrial-2; % Possible no. of easy distractors in a trial

% In Experiment 2 we have one target in each trial, which can either be easy
% or hard. This target is presented with a varying number of distractors from
% the same family
epar.trials.disBlocks = [];
if epar.expNo == 2

    % No. of targets per trial
    % Easy/hard targets are shown in seperate trials, thefore, a miniblock
    % in Experiment 2 has double the size compared to Experiment 3
    epar.trials.disBlocks(1:length(distractorNoS)*2, 1) = 1;

    % No. of easy/hard distractors
    % Distractors belong to the same family as the target
    epar.trials.disBlocks(:, 2) = [distractorNoS'; zeros(length(distractorNoS), 1)];
    epar.trials.disBlocks(:, 3) = flipud(epar.trials.disBlocks(:, 2));

    % Target type (easy or hard)
    epar.trials.disBlocks(1:length(distractorNoS), 4)     = 1;
    epar.trials.disBlocks(length(distractorNoS)+1:end, 4) = 2;

% In Experiment 3 we show both targets in each trial. Additionally, we
% present a varying number of easy and hard distractors
elseif epar.expNo == 3

    % No. of targets per trial
    % In each trial of Experiment 3, both targets are shown
    epar.trials.disBlocks(1:length(distractorNoS), 1) = 2; 

    % No. of easy/hard distractors
    % In each trial of Experiment 3, we show "maxNoOfStimuliInTrial - 
    % CurrentNoOfEasyDistractors" hard distractors
    epar.trials.disBlocks(:, 2) = distractorNoS';
    epar.trials.disBlocks(:, 3) = noOfStimInTrial - ...
                                  epar.trials.disBlocks(:, 1) - ...
                                  epar.trials.disBlocks(:, 2);

end

% Generate an arbitrary number shuffled miniblocks
epar.trials.disBlocksRand = [];
for p = 1:1000

    % Set how many combinations easy/hard distractor no. we have
    disBlocks_idx = length(epar.trials.disBlocks);

    % Shuffle miniblock
    disBlocks_idx = randperm(disBlocks_idx)'; 

    % Append the shuffled version to the empty array
    epar.trials.disBlocksRand = [epar.trials.disBlocksRand; ...
                                 epar.trials.disBlocks(disBlocks_idx, :)];

end


%% Misc. settings
% Set trial number
epar.trial.num  = epar.trialNo;

% Set which stimulus (easy/hard) will be shown in a given trial
% In Experiment 3, both targets are shown in each trial
if epar.expNo == 2

    epar.diff = epar.trials.disBlocksRand(:, 4);

elseif epar.expNo == 3

    epar.diff = NaN(epar.trialNo, 1);

end

epar.trials.targ(1:epar.trial.num)          = epar.targ;  % No. of targets per trial
epar.trials.stairSteps(1:epar.trial.num, 1) = epar.diff2; % Difficulty (easy target)
epar.trials.stairSteps(1:epar.trial.num, 2) = epar.diff3; % Difficulty (hard target)