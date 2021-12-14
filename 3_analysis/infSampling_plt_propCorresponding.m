function infSampling_plt_propCorresponding(corr_dat, lat_dat, plt)

    % Plots proportion trials for which last saccaded and responded on
    % target corresponded and proportion trials in which they did not as
    % well as latency distributions for trials with the last gaze shift to
    % the background and to the target
    % Input
    % corr_dat: matrix with proportions for congruent (:, 1) and
    %           incongruent (:, 2) trials
    % lat_dat:  matrix with latencies when last gaze shift landed on
    %           background (:, 1) and when it landed on target (:, 2)
    % plt:
    % Output
    % --

    %% Trials with vs. trials without correspondence
    subplot(1, 2, 1)
    plot(corr_dat(:, 1), corr_dat(:, 2), ...
         'o', ...
         'MarkerSize',      plt.size.mrk_ss, ...
         'MarkerFaceColor', plt.color.c1, ...
         'MarkerEdgeColor', plt.color.white)
    line([0 1], [0 1], ...
         'LineWidth', plt.lw.thick, ...
         'LineStyle', '--', ...
         'Color',     plt.color.c2)
    axis([0 1 0 1], 'square')
    xticks(0:0.25:1)
    yticks(0:0.25:1)
    xlabel('Proportion trials congruent')
    ylabel('Proportion trials incongruent')
    box off


    %% Response time for trials with last gaze shift to background vs. trials with last gaze shift to target
    % Response time = time between gaze shift offset (either on target or 
    % background) and response
    subplot(1, 2, 2)
    histogram(lat_dat(:, 1), ...
              'FaceColor', plt.color.r2, ...
              'EdgeColor', plt.color.white)
    hold on
    histogram(lat_dat(:, 2), ...
              'FaceColor', plt.color.b2, ...
              'EdgeColor', plt.color.white)
    hold off
    axis([0 max(lat_dat(:)) 0 600], 'square')
    box off
    xlabel('Response latency [ms]')
    ylabel('N')
    lgnd = legend({'On background'; 'On target'});
    lgnd.Title.String = 'Last gaze shift';
    legend box off

end