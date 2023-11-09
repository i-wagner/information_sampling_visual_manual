xclear all; close all; clc


%% ADDITIONAL ANALYSIS COMPARING EYE TRACKING AND TABLET RESULTS
%% Settings
exper.name.root = '/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual';
exper.name.plt = strcat(exper.name.root, '/', '4_figures/');


%% Load data
dataTablet = load(['/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual/2_data/', 'dataTablet.mat']);
dataEye = load(['/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual/2_data/', 'dataEye.mat']);


%% Compare perceptual performance
perfEasy = [dataEye.perf.hitrates(:, 1, 2), ...
            dataTablet.perf.hitrates(:, 1, 2)];
perfDiff = [dataEye.perf.hitrates(:, 1, 3), ...
            dataTablet.perf.hitrates(:, 1, 3)];
dataToPlot = cat(3, perfEasy, perfDiff);
tit = {'Easy target', 'Difficult target'};

fig.h = figure;
tiledlayout(1, 2);
for sp = 1:size(dataToPlot, 3)

    nexttile
    hold on
    line([0, 1], [0, 1]);
    line([0, 1], [0.50, 0.50]);
    line([0.50, 0.50], [0, 1]);
    plot(dataToPlot(:,1,sp), dataToPlot(:,2,sp), ...
         'o')
    plotMean(dataToPlot(:,1,sp), dataToPlot(:,2,sp), [0, 0, 0]);
    hold off
    axis([0, 1, 0, 1], 'square');
    xticks(0:0.25:1);
    yticks(0:0.25:1);
    xlabel('Proportion correct [eye]');
    ylabel('Proportion correct [tablet]');
    title(tit{sp});
    box off

end
opt.size = [45, 15];
opt.imgname = strcat(exper.name.plt , 'perceptualPerformance');
opt.save = 1;
prepareFigure(fig.h, opt)
close; clear fig opt perfEasy perfDiff dataToPlot tit


%% Compare noise in probabilistic generative model
dataToPlot = cat(3, dataEye.model.freeParameter{2}, dataTablet.model.freeParameter{2});
tit = {'Fixation noise', 'Decision noise'};

fig.h = figure;
tiledlayout(1, 2);
for sp = 1:size(dataToPlot, 2)

    nexttile
    hold on
    line([0, 2], [0, 2]);
    plot(dataToPlot(:,sp,1), dataToPlot(:,sp,2), ...
         'o')
    plotMean(dataToPlot(:,sp,1), dataToPlot(:,sp,2), [0, 0, 0]);
    hold off
    axis([0, 2, 0, 2], 'square');
    xticks(0:0.50:2);
    yticks(0:0.50:2);
    xlabel('SD of noise [eye]');
    ylabel('SD of noise [tablet]');
    title(tit{sp});
    box off

end
opt.size = [45, 15];
opt.imgname = strcat(exper.name.plt , 'modelParameters');
opt.save = 1;
prepareFigure(fig.h, opt)
close; clear fig opt dataToPlot tit


%%
propGsTablet = squeeze(mean(infSampling_avgPropSacc(dataTablet.sacc.propGs.onChosen_trialBegin(:, 2), 1), 3, 'omitnan'));
propGsEye = squeeze(mean(infSampling_avgPropSacc(dataEye.sacc.propGs.onChosen_trialBegin(:, 2), 1), 3, 'omitnan'));