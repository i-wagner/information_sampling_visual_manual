function pltLatencies(dataVisual, dataManual, plt, opt)

    %% Settings
    % Mean latency time (averaged over all trials of a participant)
    % (:,1): averaged over both difficulties
    % (:,2): averaged for easy
    % (:,3): averaged for difficult
    latencyFirstGs_visual = dataVisual.sacc.latency.firstGs;
    latencyFirstGs_manual = dataManual.sacc.latency.firstGs;
    
    axisLimits = [100, 610];
    plotTitle = {'Both'; 'Easy'; 'Difficult'};

    %% Plot
    figureHandle = figure;
    tiledlayout(2, 3);
    for c = 1:2 % Condition
        for p = 1:3 % Panel
            thisData = [latencyFirstGs_visual(:,p,c), ...
                        latencyFirstGs_manual(:,p,c)];
        
            axisHandle = nexttile;
            line([axisLimits(1), axisLimits(2)], ...
                 [axisLimits(1), axisLimits(2)], ...
                 'LineStyle',        '--', ...
                 'LineWidth',        plt.line.widthThin, ...
                 'Color',            plt.color.gray(3,:), ...
                 'HandleVisibility', 'off')
            hold on
            plot(thisData(:,1), thisData(:,2), ...
                 'o', ...
                 'MarkerSize',      plt.marker.sizeSmall, ...
                 'MarkerFaceColor', plt.color.gray(2,:), ...
                 'MarkerEdgeColor', plt.color.white, ...
                 'LineWidth',       plt.line.widthThin);
            set(axisHandle, ...
                'XColor', plt.color.green(1,:), ...
                'YColor', plt.color.purple(1,:));
            [~, ~, meanHandle] = plotMean(thisData(:,1), thisData(:,2), plt.color.black);
            set(meanHandle, ...
                'MarkerSize',      plt.marker.sizeLarge, ...
                'MarkerFaceColor', plt.color.black, ...
                'MarkerEdgeColor', 'none', ...
                'LineWidth',       plt.line.widthThin);
            set(meanHandle(2:end), ...
                'HandleVisibility', 'off');
            hold off
            axis([axisLimits(1), axisLimits(2), axisLimits(1), axisLimits(2)], 'square');
            xticks(axisLimits(1):100:axisLimits(2));
            yticks(axisLimits(1):100:axisLimits(2));
            xlabel('Latency 1st sacc. [ms]');
            ylabel('Latency 1st mov. [ms]');
            if c == 1
                title(plotTitle{p});
            end
            box off
        end
    end
    
    %% Save plot
    opt.size = [55, 35];
    opt.imgname = [plt.figurePath, 'figure5.png'];
    opt.save = 1;
    prepareFigure(figureHandle, opt);
    close

end