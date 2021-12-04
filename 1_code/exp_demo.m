%% Clear workspace and add path
clear variables;
close all;
clc;
addpath('_lib', '_lib\additionalInPolygonStuff', '_lib\coder\lib');


%% General settings
% Set the random number generator
rng('default');
rng('shuffle');

% Set experiment, task and subject number
epar.block   = 99;
epar.subject = input('Subject:');
epar.expNo   = input('Experiment:'); % 2 == One target, variable distractors
                                     % 3 == Two targets, variable distractors

% Throw an error if Experiment 1 was started accidentally
if epar.expNo == 1

    error('We are not doing Experiment 1 anymore; please proceed with Experiment 2!');

end

% Load settings
exp_settings;
eval(sprintf('exp_settings_1'));

% Create folder for participant's data
epar.exp_path = sprintf('%s/e%dv%db%d', epar.save_path, epar.expNo, ...
                        epar.subject, epar.block');

% If folder exists, delete and create a new one
if isdir(epar.exp_path)

    rmdir([epar.exp_path], 's');
    mkdir([epar.exp_path]);

else

    mkdir([epar.exp_path]);

end

% Create name of .log file
epar.log_file = sprintf('%s/e%dv%db%d.log', epar.exp_path, epar.expNo, ...
                        epar.subject, epar.block);


%% Number of demo trials
epar.trial.num = 10;


%% Monitor settings
epar = exp_mon_init(epar);


%% Initialize eye link
if epar.EL

    el = exp_el_init(epar);

else

    el = NaN;

end
epar.CALIB = 1;


%% Show instructions screen
[epar, el] = exp_trial_init(epar, el, 1);
exp_instruction(epar);


%% Calibrate eye link
if epar.EL

    result = EyelinkDoTrackerSetup(el);

    if result == el.TERMINATE_KEY

        return

    end

    Eyelink('message', 'Block_Start');
    Eyelink('WaitForModeReady', 500);

end


%% Present trials
epar = exp_self_start(epar);
for t = 1:epar.trial.num

    % Generate trial
    [epar, el] = exp_trial_init(epar, el, t);

    % Start data recording
    if epar.EL

        exp_el_start(el, t, epar.x_center, epar.y_center);

    end

    % Present trial
    epar = exp_trial_show(epar, t);

    % Perceptual judgement
    epar = exp_trial_response(epar, el, t);

    % Stop data recording
    if epar.EL

        WaitSecs(0.05);
        Eyelink('StopRecording');
        error = Eyelink('CheckRecording');
        fprintf('Stop Recording: %d; ', error);
        Eyelink('SetOfflineMode');
        WaitSecs(0.05);
        playbackResult(t) = EL2_playback(epar.eye_name);

    end

    % Check eye movement recording
    epar = exp_trial_eval(epar);

    % Update score and show feedback screen
    epar = exp_sc_update(epar, t);

    % Update log file
    exp_trial_save(epar, t);

end


%% Finish the experiment
Screen('Close', epar.stim.txt_disp);
exp_el_exit(epar);
exp_mon_exit(epar);


%% Display a subject's score in console
strcat('This participants score is: ', num2str(epar.score))