clear all; close all; clc

%% Init
initFig;
folder.fig = strcat(folder.root, "5_outreach/manuscript/figures/fig3");

%% Panels A&B: choice curves of representative participants
% Number of participants to use as representative participants
subjectOfInterest = [9, 9];
conditionsOfInterest = [2, 4]; % Only double-target conditions

idx.intercept = 1;
idx.slope = 2;
chancePerformance = 0.50;
nDistractorsBalanced = 4;

axisLimits = [[-1, 9]; [0, 1.15]];
lineLimitsHorizontal = [axisLimits(1,:)', [4; 4]];
lineLimitsVertical = [[0.50; 0.50], [0; axisLimits(2,2)]];
yLabels = {'Prop. choices easy [visual]', 'Prop. choices easy [manual]'};

hFig = figure;
tiledlayout(1, 3);
for c = 1:numel(conditionsOfInterest) % Condition
    % For yPredicted:
    % subtract the number of distractors at which both sets have an equal
    % size for generating model predictions, because this was also done
    % when estimating regression parameters. At the end, addd chance
    % performance to the predicted values, again, because chance
    % performance was initially subtracted when estimating regression 
    % parameters. For an explanation why this was done, see the manuscript
    % or the function, where parameters are fitted ("fitRegression")
    intercept = data.choice.regressionFit(subjectOfInterest(c),idx.intercept,conditionsOfInterest(c));
    slope = data.choice.regressionFit(subjectOfInterest(c),idx.slope,conditionsOfInterest(c));
    x = (0:1:8)';
    yEmpirical = ...
        data.choice.target.proportionEasy(subjectOfInterest(c),:,conditionsOfInterest(c));
    yIdealObserver = idealObserver.proChoices.easy(subjectOfInterest(c),:,conditionsOfInterest(c));
    yPredicted = (intercept + slope .* (x - nDistractorsBalanced)) + chancePerformance;

    if c == 1 % Visual search experiment
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
         'Color', plt.color.gray(3,:));
    hold off
    axis([axisLimits(1,:), axisLimits(2,:)], 'square')
    xticks((axisLimits(1,1)+1):2:(axisLimits(1,2)-1))
    yticks(axisLimits(2,1):0.25:axisLimits(2,2))
    xlabel('# easy distractors');
    ylabel(yLabels(c));
    box off
    if c == 1
        legend({'Ideal obs.', 'Data', 'Regression'}, ...
                'Location', 'Southwest');
        legend box off
    end

    if checkAxLim(axisLimits(1,:), x) | ...
       checkAxLim(axisLimits(2,:), [yIdealObserver(:); yEmpirical(:); yPredicted(:)])
        error("Current axis limits result in values being cut-off!");
    end
end

%% Panel C: regression parameter all participants
intercepts = squeeze(data.choice.regressionFit(:,idx.intercept,conditionsOfInterest));
slopes = squeeze(data.choice.regressionFit(:,idx.slope,conditionsOfInterest));
regPar = cat(3, intercepts, slopes);
nParameter = size(regPar, 3);

axisLimits = [min(regPar(:)), max(regPar(:))];
plotMarker = {'o', 's'};
lineHorizontal = [axisLimits', axisLimits', [0; 0]];
lineVertical = [axisLimits', [0; 0], axisLimits'];

nexttile;
line(lineHorizontal, lineVertical, ...
     'LineStyle', '-', ...
     'LineWidth', plt.line.widthThin, ...
     'Color', plt.color.black, ...
     'HandleVisibility', 'off');
hold on
for p = 1:nParameter % Parameter
    plot(regPar(:,1,p), regPar(:,2,p), ...
         'Marker', plotMarker{p}, ...
         'MarkerSize', plt.marker.sizeSmall, ...
         'MarkerFaceColor', plt.color.gray(2,:), ...
         'MarkerEdgeColor', plt.color.white, ...
         'LineStyle', 'none', ...
         'LineWidth', plt.line.widthThin, ...
         'HandleVisibility', 'off');
    [~, ~, meanHandle] = plotMean(regPar(:,1,p), regPar(:,2,p), plt.color.black);
    set(meanHandle, ...
        'MarkerSize', plt.marker.sizeLarge, ...
        'MarkerFaceColor', plt.color.black, ...
        'MarkerEdgeColor', 'none', ...
        'LineWidth', plt.line.widthThin);
    set(meanHandle(1), ...
        'Marker', plotMarker{p});
    set(meanHandle(2:end), ...
        'HandleVisibility', 'off')
end
hold off
axis([axisLimits, axisLimits], 'square')
xticks(-0.10:0.10:0.50);
yticks(-0.10:0.10:0.50);
xlabel('Visual search')
ylabel('Manual search')
legend({'Intercept [diff.]'; 'Slope [costs]'}, ...
        'Position', [0.61, 0.19, 0.50, 0.14]);
legend box off
box off

if checkAxLim(axisLimits, regPar(:))
    error("Current axis limits result in values being cut-off!");
end

sublabel([], -130, -30);
opt.size = [50, 15];
opt.imgname = folder.fig;
opt.save = true;
prepareFigure(hFig, opt);
close