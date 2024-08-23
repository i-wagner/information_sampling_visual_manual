clear all; close all; clc

%% Init
initFig;
folder.fig = [strcat(folder.root, "5_outreach/manuscript/figures/figSupp3"), ...
              strcat(folder.root, "5_outreach/manuscript/figures/figSupp4")];

%% Plot
predDat = cat(3, ...
              probabilisticModel.pred.visual.propChoicesEasy, ...
              probabilisticModel.pred.manual.propChoicesEasy); 
subjectOfInterest = find(~isnan(predDat(:,1,1)));
conditionsOfInterest = [2, 4];
nSubjects = numel(subjectOfInterest);
nConditions = numel(conditionsOfInterest);

x = (0:1:8)';
axisLimits = [[(x(1) - 1), (x(end) + 1)]; ...
              [0, 1.15]];
lineLimitsHorizontal = [axisLimits(1,:)', [4; 4]];
lineLimitsVertical = [[0.50; 0.50], [0; axisLimits(2,2)]];
yLabels = {'Prop. choices easy [visual]', 'Prop. choices easy [manual]'};
for c = 1:nConditions % Conditions
    hFig = figure;
    hTile = tiledlayout(5, 4);
    for s = 1:nSubjects % Subjects
        yEmpirical = ...
            data.choice.target.proportionEasy(subjectOfInterest(s),:,conditionsOfInterest(c));
        yPredicted = predDat(subjectOfInterest(s),:,c);
    
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
        plot(x, yEmpirical, ...
             'o-', ...
             'MarkerSize', plt.marker.sizeSmall, ...
             'MarkerFaceColor', thisColor, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth', plt.line.widthThin, ...
             'Color', thisColor)
        plot(x, yPredicted, ...
             'd-', ...
             'MarkerSize', plt.marker.sizeSmall, ...
             'MarkerFaceColor', thisColor, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth', plt.line.widthThin, ...
             'Color', thisColor)
        hold off
        axis([axisLimits(1,:), axisLimits(2,:)], 'square')
        xticks((axisLimits(1,1)+1):4:(axisLimits(1,2)-1))
        yticks(axisLimits(2,1):0.50:axisLimits(2,2))
        box off

        if checkAxLim(axisLimits(2,:), ...
                      [yEmpirical(:), yPredicted(:)])
            error("Current axis limits result in values being cut-off!");
        end
    end
    xlabel(hTile, '# easy distractors', "FontSize", opt.fontSize);
    ylabel(hTile, yLabels(c), "FontSize", opt.fontSize);
    hLgd = legend({'Empirical', 'Model'});
    hLgd.Layout.Tile = 20;
    legend box off

    sublabel([], -5, -45);
    opt.size = [35, 45];
    opt.imgname = folder.fig(c);
    opt.save = true;
    prepareFigure(hFig, opt);
    close;
end