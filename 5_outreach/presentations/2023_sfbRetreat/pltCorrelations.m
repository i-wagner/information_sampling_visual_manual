function pltCorrelations(dataVisual, plt, opt)

    %% Get proportion first gaze shifts on chosen target (double-target)
    % Initially, proportion gaze shifts are stored for each participant, each
    % set-size condition, and each executed sacade seperately.
    % - In a first step, we re-arrange the data in a matrix (instead of a cell
    %   array), which stores, for each subject, each set-size condition, and
    %   each gaze shift in a trial, the proportion of gaze shifts in a trial
    %   that landed on an element from the set of the chosen target (seperately
    %   for each gaze shift that was made in a trial)
    % - In a second step, we average, for each participant, over the
    %   proportions of each set-size condition to get a point estimate for each
    %   gaze shift in a trial
    % - Finaly, we extract the proportions for the first gaze shifts for
    %   subsequent analysis
    proportionGazeShifts = dataVisual.sacc.propGs.onChosen_trialBegin(:,2);
    proportionGazeShifts = infSampling_avgPropSacc(proportionGazeShifts, []);
    proportionGazeShifts = squeeze(mean(proportionGazeShifts, 3, 'omitnan'));
    
    proportionFirstGazeShiftsOnChosen = proportionGazeShifts(:,1);
    
    %% Get latencies of first gaze shifts in trials (double-target)
    latenciesFirstGazeShift = dataVisual.sacc.latency.firstGs(:,1,2);
    latenciesFirstGazeShift = latenciesFirstGazeShift(~isnan(latenciesFirstGazeShift));
    
    %% Flag outlier
    idx = [3, 10];
    
    %% Plot
    axScaleFactor = 0.10;
    axisLimits = [min(latenciesFirstGazeShift), max(latenciesFirstGazeShift), ...
                  min(proportionFirstGazeShiftsOnChosen), max(proportionFirstGazeShiftsOnChosen)];
    axisLimits = [floor(axisLimits(1)), ceil(axisLimits(2)), ...
                  floor(axisLimits(3)), ceil(axisLimits(4))];
    axisLimits(1:2) = [axisLimits(1) - (axisLimits(1) * axScaleFactor), ...
                       axisLimits(2) + (axisLimits(2) * axScaleFactor)];
    
    close all
    figureHandle = figure;
    tiledlayout(1, 2, ...
                'TileSpacing', 'Tight', ...
                'Padding', 'Tight');
    for sp = 1:2
        x = latenciesFirstGazeShift;
        y = proportionFirstGazeShiftsOnChosen;
        if sp == 2
            x(idx) = NaN;
            y(idx) = NaN;
        end
        [r, p] = corrcoef(x, y, 'Rows', 'Complete');
    
        nexttile;
        line(axisLimits(1:2), [0.50, 0.50], ...
             'LineStyle',        '--', ...
             'LineWidth',        plt.line.widthThin, ...
             'Color',            plt.color.gray(3,:), ...
             'HandleVisibility', 'off')
        hold on
        plot(x, y, ...
             'o', ...
             'MarkerSize',      plt.marker.sizeSmall, ...
             'MarkerFaceColor', plt.color.gray(2,:), ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth',       plt.line.widthThin);
        hLsLine = lsline;
        set(hLsLine, ...
            'LineStyle', '-', ...
            'Color', plt.color.black, ...
            'LineWidth', plt.line.widthThin)
        hold off
        axis(axisLimits, 'square');
        xticks(150:150:1000);
        yticks(0:0.25:1);
        text(150, 0.95, ['r = ', num2str(round(r(1,2), 2)), ', ' ...
                         'p = ', num2str(round(p(1,2), 2))]);
        xlabel('Latency 1st gaze shifts [ms]');
        ylabel('Proportion 1st gaze shifts [on chosen]');
        legend({'Subjects', 'Least-squares line'}, ...
                'Location', 'SouthEast');
        legend box off
    end
    opt.size = [40, 20];
    opt.imgname = [plt.figurePath, 'figure6.png'];
    opt.save = 1;
    prepareFigure(figureHandle, opt);
    close;

end