clear all; close all; clc;

%% Plot visuals
vssPoster_visuals;
plt.figurePath = ['/Users/ilja/Dropbox/12_work/' ...
                  'mr_informationSamplingVisualManual/5_outreach/' ...
                  'presentations/2023_sfbRetreat/'];

%% Load data
dataPath = ['/Users/ilja/Dropbox/12_work/' ...
            'mr_informationSamplingVisualManual/5_outreach/' ...
            'presentations/2023_sfbRetreat/'];
dataVisual = load([dataPath, 'dataEye.mat']);
dataManual = load([dataPath, 'dataTablet.mat']);

%% Plot
pltLatencies(dataVisual, dataManual, plt, opt);
pltCorrelations(dataVisual, plt, opt);