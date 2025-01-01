%% Experiment settings
% Visual search
% 2 = single-target
% 3 = double-target
% 
% Manual search
% 2 = single-target
% 3 = double-target
exper.num.CONDITIONS = 2:5;
exper.num.SUBJECTS = 1:24;

exper.n.CONDITIONS = numel(exper.num.CONDITIONS);
exper.n.SUBJECTS = numel(exper.num.SUBJECTS);
if exper.n.CONDITIONS < 2 | diff(exper.num.CONDITIONS) ~= 1
    error(['Single-target and double-target condition must be analysed' ...
           'together, please ensure that condition numbers are consecutive']); 
end

exper.availableTime = 6.50; % Minutes; read: six and a half minutes
exper.payoff = [2, -2]; % Eurocents

%% Stimuli
% Each stimulus on the screen gets an AOI, which is centered at the
% stimulus location and which has a fixed diameter
exper.stimulus.size.PX = 49;
% stim.diameter.deg = stim.diameter.px * screen.xPIX2DEG;
exper.stimulus.aoi.diameter.DVA = 3;
exper.stimulus.aoi.radius.DVA = exper.stimulus.aoi.diameter.DVA / 2;

% Stimuli in a trial will get assigned an identifier, allowing us to put it
% into one of five categories
exper.stimulus.id.target.EASY = 1;
exper.stimulus.id.target.DIFFICULT = 2;
exper.stimulus.id.distractor.EASY = 3;
exper.stimulus.id.distractor.DIFFICULT = 4;
exper.stimulus.id.BACKGROUND = 666;

%% Fixation cross
exper.fixation.location.x.DVA = 0;
exper.fixation.location.y.DVA = -9.50; % Below screen center
exper.fixation.location.x.PX = 960;
exper.fixation.location.y.PX = 912;

%% Path settings
% For figures:
% path has this weird format so we can ust the condition number to index to
% the proper path for the right effector
exper.path.ROOT = ['/Users/ilja/Library/CloudStorage/' ...
                   'GoogleDrive-ilja.wagner1307@gmail.com/My Drive/' ...
                   'mr_informationSamplingVisualManual/'];
exper.path.DATA = strcat(exper.path.ROOT, '2_data/');
exper.path.ANALYSIS = strcat(exper.path.ROOT, '3_analysis/');
exper.path.figures.singleSubjects = ...
    {strcat(exper.path.ROOT, '4_figures/eye/singleSubject/single_'), ...
     strcat(exper.path.ROOT, '4_figures/eye/singleSubject/double_'), ...
     strcat(exper.path.ROOT, '4_figures/tablet/singleSubject/single_'), ...
     strcat(exper.path.ROOT, '4_figures/tablet/singleSubject/double_')};