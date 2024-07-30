clear all; close all; clc

%% Helper function
function [dat_mean, dat_cis] = ciForProportion(data)

    % Perform arcine transformation on data, and calculate means as well as
    % confidence intervals based on transformed data
    %
    % NOTE 1:
    % this avoids confidence intervals that are greater then one
    %
    % NOTE 2:
    % outputs are calculated based on arcsine-d values, which are then
    % backtransformed. Thus,those means do not necessarily correspond to
    % the means that we get if calculating the average over raw data
    %
    % NOTE 3:
    % confidence intervals might be asymmetric
    %
    % Input
    % data:
    % matrix; should be in wide format, i.e., rows are participants,
    % columns are variables
    %
    % Output
    % dat_mean:
    % vector; mean for each variable, calculated based on arcsin values,
    % then backtransformed into original scale
    % 
    % dat_cis:
    % vector; confidence intervals for each variable, calculated based on 
    % arcsin values, then backtransformed into original scale

    dat_arcsine = asin(sqrt(data));
    dat_mean_arcsine = mean(dat_arcsine, 1, 'omitnan');
    dat_cis_arcsine = ci_mean(dat_arcsine);
    dat_cis_arcsine = [dat_mean_arcsine - dat_cis_arcsine; ...
                       dat_mean_arcsine + dat_cis_arcsine];
    dat_mean = sin(dat_mean_arcsine).^2;
    dat_cis = sin(dat_cis_arcsine).^2;
    dat_cis = dat_cis - dat_mean;
end

%% Init
initFig;
folder.fig = strcat(folder.root, "5_outreach/manuscript/figures/fig6");

%% Panels A&B: predicted vs. empirical avg. proportion choices for easy target
idx.doubleVisual = 2;
idx.doubleManual = 4;
yLabels = ["Prop. choices easy [visual]", ...
           "Prop. choices easy [manual]"];
x = 0:1:8;
chancePerformance = 0.50;
nDistractorsEqualSetSize = 4;
tiles = [1, 3];

hFig = figure;
tiledlayout(2, 2);
for p = 1:2 % Panel
    if p == 1 % Visual search experiment
        plt.color.condition = plt.color.green;

        emp = data.choice.target.proportionEasy(:,:,idx.doubleVisual);
        pred = probabilisticModel.pred.visual.propChoicesEasy;
    else
        plt.color.condition = plt.color.purple;

        emp = data.choice.target.proportionEasy(:,:,idx.doubleManual);
        pred = probabilisticModel.pred.manual.propChoicesEasy;
    end

    % Arcsine transformation to avoid CIs greater than one
    [emp_mean, emp_cis] = ciForProportion(emp);
    [pred_mean, pred_cis] = ciForProportion(pred);

    nexttile(tiles(p));
    line([[(x(1) - 1); (x(end) + 1)], ...
          [nDistractorsEqualSetSize; nDistractorsEqualSetSize]], ...
         [[chancePerformance; chancePerformance], ...
          [0; 1]], ...
         'LineStyle', '-', ...
         'LineWidth', plt.line.widthThin, ...
         'Color', plt.color.black, ...
         'HandleVisibility', 'off');
    hold on
    errorbar((x-0.25), emp_mean, emp_cis, ...
             'o', ...
             'MarkerSize', plt.marker.sizeLarge, ...
             'MarkerFaceColor', plt.color.condition(1,:), ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth', plt.line.widthThin, ...
             'CapSize', 0, ...
             'Color', plt.color.condition(1,:), ...
             'HandleVisibility', 'off')
    errorbar((x+0.25), pred_mean, pred_cis, ...
             'd', ...
             'MarkerSize', plt.marker.sizeLarge, ...
             'MarkerFaceColor', plt.color.condition(2,:), ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth', plt.line.widthThin, ...
             'CapSize', 0, ...
             'Color', plt.color.condition(2,:), ...
             'HandleVisibility', 'off')
    % Plot some data at a non-sensical location and use this for the
    % legend; makes it easier to control visual appearance if legend
    % marker
    plot(999, 999, ...
         'o', ...
         'MarkerSize', plt.marker.sizeLarge, ...
         'MarkerFaceColor', plt.color.condition(1,:), ...
         'MarkerEdgeColor', 'None')
    plot(999, 999, ...
         'd', ...
         'MarkerSize', plt.marker.sizeLarge, ...
         'MarkerFaceColor', plt.color.condition(2,:), ...
         'MarkerEdgeColor', 'None')
    hold off
    axis([(x(1) - 1), (x(end) + 1), 0, 1]);
    xlabel("# easy distractors");
    ylabel(yLabels(p));
    xticks(x(1):1:x(end));
    yticks(0:0.25:1);
    legend({"Empirical", "Model"}, 'Location', 'SouthWest');
    legend box off
    box off
end

%% Panels C&D: predicted vs. empirical avg. proportion fixations on chosen set
xLabels = ["Emp. prop. mov. chosen [visual]", ...
           "Emp. prop. mov. chosen [manual]"];
yLabels = ["Pred. prop. mov. chosen [visual]", ...
           "Pred. prop. mov. chosen [manual]"];
tiles = [2, 4];
for p = 1:2 % Panel
    if p == 1 % Visual search experiment
        emp = mean(data.fixations.propFixOnChosenModelEval(:,:,idx.doubleVisual), 2, 'omitnan');
        pred = mean(probabilisticModel.pred.visual.propFixChosen, 2, 'omitnan');
    else
        emp = mean(data.fixations.propFixOnChosenModelEval(:,:,idx.doubleManual), 2, 'omitnan');
        pred = mean(probabilisticModel.pred.manual.propFixChosen, 2, 'omitnan');
    end

    nexttile(tiles(p));
    line([0, 1], [0, 1], ...
         'LineStyle', '-', ...
         'LineWidth', plt.line.widthThin, ...
         'Color', plt.color.black, ...
         'HandleVisibility', 'off');
    hold on
    plot(emp, pred, ...
         'o', ...
         'MarkerSize', plt.marker.sizeSmall, ...
         'MarkerFaceColor', plt.color.gray(2,:), ...
         'MarkerEdgeColor', plt.color.white, ...
         'LineWidth', plt.line.widthThin, ...
         'Color', plt.color.black)
    errorbar(mean(emp, 1, 'omitnan'), mean(pred, 1, 'omitnan'), ...
             ci_mean(pred), ci_mean(pred), ...
             ci_mean(emp), ci_mean(emp), ...
             'o', ...
             'MarkerSize', plt.marker.sizeLarge, ...
             'MarkerFaceColor', plt.color.black, ...
             'MarkerEdgeColor', 'none', ...
             'LineWidth', plt.line.widthThin, ...
             'CapSize', 0, ...
             'Color', plt.color.black, ...
             'HandleVisibility', 'off')
    [~, ~, h] = plotMean(emp, pred, plt.color.black);
    set(h(4), 'LineWidth', plt.line.widthThin);
    hold off
    axis([0, 1, 0, 1], 'square');
    xlabel(xLabels(p));
    ylabel(yLabels(p));
    xticks(0:0.25:1);
    yticks(0:0.25:1);
    box off
end

sublabel([], -20, -40);
opt.size = [45, 35];
opt.imgname = folder.fig;
opt.save = true;
prepareFigure(hFig, opt);
close;