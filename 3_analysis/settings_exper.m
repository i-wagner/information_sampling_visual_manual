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

%% Path settings
% For figures:
% path has this weird format so we can ust the condition number to index to
% the proper path for the right effector
exper.path.ROOT = ['/Users/ilja/Dropbox/12_work/' ...
                   'mr_informationSamplingVisualManual/'];
exper.path.DATA = strcat(exper.path.ROOT, '2_data');
exper.path.ANALYSIS = strcat(exper.path.ROOT, '3_analysis');
exper.path.FIGURES = {{}, ...
                      strcat(exper.path.ROOT, '4_figures/eye'), ...
                      strcat(exper.path.ROOT, '4_figures/eye'), ...
                      strcat(exper.path.ROOT, '4_figures/tablet'), ...
                      strcat(exper.path.ROOT, '4_figures/tablet')};

addpath(exper.path.ANALYSIS);