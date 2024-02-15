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
exper.stimulus.id.target.DIFFICULT = 3;
exper.stimulus.id.distractor.EASY = 2;
exper.stimulus.id.distractor.DIFFICULT = 4;
exper.stimulus.id.BACKGROUND = 666;

%% Fixation cross
exper.fixation.tolerance.DVA = 1.50;
exper.fixation.location.x.DVA = 0;
exper.fixation.location.y.DVA = 9.50;
% exper.fixation.location.x.PX = ...
%     round((exper.fixation.location.x.DVA / screen.pix2deg.X) + ...
%           screen.center.x.PX);
% exper.fixation.location.y.PX = ...
%     round((exper.fixation.location.y.DVA / screen.pix2deg.Y) + ...
%           screen.center.y.PX);

%% Path settings
% For figures:
% path has this weird format so we can ust the condition number to index to
% the proper path for the right effector
exper.path.ROOT = ['/Users/ilja/Dropbox/12_work/' ...
                   'mr_informationSamplingVisualManual/'];
exper.path.DATA = strcat(exper.path.ROOT, '2_data');
exper.path.ANALYSIS = strcat(exper.path.ROOT, '3_analysis');
exper.path.FIGURES = {{}, ...
                      strcat(exper.path.ROOT, '4_figures/eye/single_'), ...
                      strcat(exper.path.ROOT, '4_figures/eye/double_'), ...
                      strcat(exper.path.ROOT, '4_figures/tablet/single_'), ...
                      strcat(exper.path.ROOT, '4_figures/tablet/double_')};

addpath(exper.path.ANALYSIS);