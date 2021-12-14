function infSampling_plt_latencies(dat, lab_x, lab_y, lab_title, ax_fac, plt)

    % Plot median latencies
    % Input
    % dat:       array with data to plot. Different pages correspond to
    %            different data types that we want to show in seperate
    %            subplots. Columns on each page are data for x- (:, 1) and
    %            y-axis (:, 2)
    % lab_x:     x-axis labels for subplots
    % lab_y:     y-axis labels for subplots
    % lab_title: titles for subplots
    % ax_fac:    factor by which we extend the axes
    % plt:       structure with general plot settings
    % Output
    % --

    %% Set axis limits
    ax_scale = [round(min(dat(:)) - (min(dat(:)) * ax_fac)) ...
                round(max(dat(:)) + (max(dat(:)) * ax_fac))];


    %% Create plot
    plot(dat(:, 1), dat(:, 2), ...                   Single subject data
         'o', ...
         'MarkerSize',      plt.size.mrk_ss, ...
         'MarkerFaceColor', plt.color.c1, ...
         'MarkerEdgeColor', plt.color.white)
    hold on
    [~, ~, p_h] = plotMean(dat(:, 1), dat(:, 2), ... Mean data
                           plt.color.black);
    set(p_h(1), ...
        'MarkerSize',      plt.size.mrk_mean, ...
        'MarkerEdgeColor', plt.color.white)
    set(p_h(2:4), ...
        'LineWidth', plt.lw.thick)
    hold off
    l_h = line([ax_scale(1) ax_scale(2)], [ax_scale(1) ax_scale(2)], ...
               'LineWidth', plt.lw.thick, ...
               'Color',     plt.color.c2, ...
               'LineStyle', '--');
    uistack(l_h, 'bottom')
    xlabel(lab_x)
    ylabel(lab_y)
    axis([ax_scale ax_scale], 'square')
    xticks(0:20:ax_scale(2))
    yticks(0:20:ax_scale(2))
    title(lab_title)
    box off

end