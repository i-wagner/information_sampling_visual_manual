clear all; close all; clc;

%% Plot visuals
vssPoster_visuals

%% Load data
dataVisual = load('/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual/2_data/dataEye.mat');
dataManual = load('/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual/2_data/dataTablet.mat');

% dataVisual = load('/Users/i/Dropbox/12_work/mr_informationSamplingVisualManual/2_data/dataEye.mat');
% dataManual = load('/Users/i/Dropbox/12_work/mr_informationSamplingVisualManual/2_data/dataTablet.mat');

%% Plot
vssPoster_fig1(dataVisual, dataManual, plt, opt);
% vssPoster_fig2(dataVisual, dataManual, plt, opt);
% vssPoster_fig3(dataVisual, dataManual, plt, opt);
% vssPoster_fig4(dataVisual, dataManual, plt, opt);