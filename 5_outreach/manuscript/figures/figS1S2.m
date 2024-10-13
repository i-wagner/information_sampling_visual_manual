clear all; close all; clc

%% Init
initFig;
folder.fig = [strcat(folder.root, "5_outreach/manuscript/figures/figSupp1"), ...
              strcat(folder.root, "5_outreach/manuscript/figures/figSupp2")];

%% Plot
subjectOfInterest = find(~isnan(data.choice.sigmoidFit(:,1,1)));
conditionsOfInterest = [2, 4];
nSubjects = numel(subjectOfInterest);
nConditions = numel(conditionsOfInterest);

idx.means = 1;
idx.sds = 2;
chancePerformance = 0.50;
nDistractorsBalanced = 4;

x = (0:1:8)';
axisLimits = [[(x(1) - 1), (x(end) + 1)]; ...
              [0, 1.15]];
lineLimitsHorizontal = [axisLimits(1,:)', [4; 4]];
lineLimitsVertical = [[0.50; 0.50], [0; axisLimits(2,2)]];
yLabels = {'Prop. choices easy [oculomotor]', 'Prop. choices easy [manual]'};
for c = 1:nConditions % Conditions
    hFig = figure;
    hTile = tiledlayout(5, 4);
    for s = 1:nSubjects % Subjects
        mean = data.choice.sigmoidFit(subjectOfInterest(s),idx.means,conditionsOfInterest(c));
        std = data.choice.sigmoidFit(subjectOfInterest(s),idx.sds,conditionsOfInterest(c));
        yEmpirical = ...
            data.choice.target.proportionEasy(subjectOfInterest(s),:,conditionsOfInterest(c));
        yIdealObserver = idealObserver.proChoices.easy(subjectOfInterest(s),:,conditionsOfInterest(c));
        if sign(std) == -1
            yPredicted = cdf('Normal', x, mean, abs(std));
        else
            yPredicted = 1 - cdf('Normal', x, mean, std);
        end

        if c == 1
            thisColor = plt.color.green(2,:);
        elseif c == 2
            thisColor = plt.color.purple(2,:);
        end
    
        nexttile;
        line(lineLimitsHorizontal, lineLimitsVertical, ...
             'LineStyle', '-', ...
             'LineWidth', plt.line.widthThin, ...
             'Color', plt.color.black, ...
             'HandleVisibility', 'off');
        hold on
        plot(x, yIdealObserver, ...
             ':', ...
             'MarkerSize', plt.marker.sizeSmall, ...
             'MarkerFaceColor', thisColor, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth', plt.line.widthThin, ...
             'Color', thisColor)
        plot(x, yEmpirical, ...
             'o-', ...
             'MarkerSize', plt.marker.sizeSmall, ...
             'MarkerFaceColor', thisColor, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth', plt.line.widthThin, ...
             'Color', thisColor)
        plot(x, yPredicted, ...
             '-', ...
             'LineWidth', plt.line.widthThick, ...
             'Color', plt.color.gray(2,:));
        hold off
        axis([axisLimits(1,:), axisLimits(2,:)], 'square')
        xticks((axisLimits(1,1)+1):4:(axisLimits(1,2)-1))
        yticks(axisLimits(2,1):0.50:axisLimits(2,2))
        box off

        if checkAxLim(axisLimits(2,:), ...
                      [yIdealObserver(:), yEmpirical(:), yPredicted(:)])
            error("Current axis limits result in values being cut-off!");
        end
    end
    xlabel(hTile, '# easy distractors', "FontSize", opt.fontSize);
    ylabel(hTile, yLabels(c), "FontSize", opt.fontSize);
    hLgd = legend({'Ideal obs.', 'Data', 'Sigmoid'});
    hLgd.Layout.Tile = 20;
    legend box off

    sublabel([], -5, -45);
    opt.size = [35, 45];
    opt.imgname = folder.fig(c);
    opt.save = true;
    prepareFigure(hFig, opt);
    close;
end