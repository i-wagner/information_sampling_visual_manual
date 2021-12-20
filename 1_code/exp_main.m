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
epar.block   = 1;
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

% If folder exists, throw error, otherwise create folder
if exist(epar.exp_path, 'dir') == 7

    error('Directory already exists! Please check experiment, subject and block number!')

else

    mkdir([epar.exp_path]);

end

% Create name of .log file
epar.log_file = sprintf('%s/e%dv%db%d.log', epar.exp_path, epar.expNo, ...
                        epar.subject, epar.block);


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
while epar.timer_cum > epar.duration

    % Generate trial
    [epar, el] = exp_trial_init(epar, el, t);

    % Start data recording
    if epar.EL

        exp_el_start(el, t, epar.fixLoc_px(1), epar.fixLoc_px(2));

    end

    % Present trial
    epar = exp_trial_show(epar, t);

    % Perceptual judgement
    epar = exp_trial_response(epar, el, t);

    % Adjust the timer after a participant placed its answer
    epar.timer_cum = epar.timer_cum + (epar.time(3) - epar.time(2));

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
Screen('Close', epar.stim.txt_disp_mask);
exp_el_exit(epar);
exp_mon_exit(epar);


%% Save epar
save([epar.exp_path '/epar.mat'], 'epar');


%% Display a subject's score in console
strcat('This participants score is: ', num2str(epar.score))