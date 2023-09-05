function vssPoster_fig2(dataVisual, dataManual, plt, opt)

    %% Panel 1: proportion gaze shifts for visual search
    minSubForMean = 1;
    
    thisData = dataVisual.sacc.propGs.onChosen_trialBegin(:,2);
    thisData = infSampling_avgPropSacc(thisData, minSubForMean);
    fixationsOnChosen.visual = mean(thisData, 3, 'omitnan');
    
    thisData = dataManual.sacc.propGs.onChosen_trialBegin(:,2);
    thisData = infSampling_avgPropSacc(thisData, minSubForMean);
    fixationsOnChosen.manual = mean(thisData, 3, 'omitnan');
    
    axisLimits = [[0, 3]; [0, 1]];
    lineLimitsHorizontal = [[0; 3]];
    lineLimitsVertical = [[0.50; 0.50]];
    nDatapoints = 2;
    
    figureHandle = figure;
    tiledlayout(1, 2);
    for p = 1:2 % Panel
        if p==1
            thisData = fixationsOnChosen.visual(:,1:nDatapoints);
            thisColor = plt.color.green;
        elseif p==2
            thisData = fixationsOnChosen.manual(:,1:nDatapoints);
            thisColor = plt.color.purple;
        end
        thisMean = mean(thisData, 1, 'omitnan');
        thisCis = ci_mean(thisData);
    
        nexttile;
        line(lineLimitsHorizontal, lineLimitsVertical, ...
            'LineStyle',        '--', ...
            'LineWidth',        plt.line.widthThin, ...
            'Color',            plt.color.gray(3,:), ...
            'HandleVisibility', 'off');
        hold on
        plot(1:nDatapoints, thisData, ...
             'o-', ...
             'MarkerSize',      plt.marker.sizeSmall, ...
             'MarkerFaceColor', thisColor(2,:), ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth',       plt.line.widthThin, ...
             'Color',           thisColor(2,:))
        errorbar(linspace((1-0.25), (nDatapoints+0.25), nDatapoints), thisMean, thisCis, ...
            'o', ...
            'MarkerSize',       plt.marker.sizeLarge, ...
            'MarkerFaceColor',  thisColor(1,:), ...
            'MarkerEdgeColor',  'none', ...
            'LineWidth',        plt.line.widthThin, ...
            'CapSize',          0, ...
            'Color',            thisColor(1,:), ...
            'HandleVisibility', 'off')
        hold off
        axis([axisLimits(1,:), axisLimits(2,:)], 'square')
        xticks((axisLimits(1,1)+1):1:(axisLimits(1,2)-1))
        yticks(axisLimits(2,1):0.25:axisLimits(2,2))
        xlabel('# movement after trial start');
        ylabel('Prop. movements [chosen set]');
        box off
    end
    
    %% Save plot
    opt.size = [35, 15];
    opt.imgname = ['/Users/ilja/Dropbox/12_work/mr_informationSamplingVisualManual/' ...
                   '5_outreach/conferences/vss2023/poster/figure2.png'];
    opt.save = 1;
    prepareFigure(figureHandle, opt);
    close

end