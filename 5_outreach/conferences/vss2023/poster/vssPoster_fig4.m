function vssPoster_fig4(dataVisual, dataManual, plt, opt)

    %% Settings
    freeParameter.visual = dataVisual.model.freeParameter{2};
    freeParameter.manual = dataManual.model.freeParameter{2};
    
    axisLimits = [[0, 2]; [0, 2]];
    plotTitle = {'Fixation noise'; 'Decision noise'};
    figurePath = ['/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual/' ...
                  '5_outreach/conferences/vss2023/poster/'];
%     figurePath = ['/Users/i/Dropbox/12_work/mr_informationSamplingVisualManual/' ... ...
%                  '5_outreach/conferences/vss2023/poster/'];

    %% Plot
    figureHandle = figure;
    tiledlayout(1, 2);
    for p = 1:2 % Panel
        thisData = [freeParameter.visual(:,p), ...
                    freeParameter.manual(:,p)];
    
        axisHandle = nexttile;
        line([0, 2], [0, 2], ...
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
        hold off
        axis([axisLimits(1,:), axisLimits(2,:)], 'square')
        xticks(axisLimits(2,1):0.50:axisLimits(2,2))
        yticks(axisLimits(2,1):0.50:axisLimits(2,2))
        xlabel('Visual search')
        ylabel('Manual search')
        title(plotTitle{p});
        box off
    end
    
    %% Save plot
    opt.size = [35, 15];
    opt.imgname = [figurePath, 'figure4.png'];
    opt.save = 1;
    prepareFigure(figureHandle, opt);
    close

end