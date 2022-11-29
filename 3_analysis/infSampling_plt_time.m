function infSampling_plt_time(plt_dat_x, plt_dat_y, plt_xLab, plt_yLab, plt_title, plt)


    %% Axis limits
    ax_scale = [min([plt_dat_x; plt_dat_y]) - (min([plt_dat_x; plt_dat_y]) * 0.05) ...
                max([plt_dat_x; plt_dat_y]) + (max([plt_dat_x; plt_dat_y]) * 0.05)];


    %% Plot
    hold on
    line([0 5000], [0 5000], ...
        'Color',     plt.color.c1, ...
        'LineStyle', '--', ...
        'LineWidth', plt.lw.thick);
    plot(plt_dat_x, plt_dat_y, ...
        'o', ...
        'MarkerSize',      plt.size.mrk_ss, ...
        'MarkerFaceColor', plt.color.o2, ...
        'MarkerEdgeColor', plt.color.white, ...
        'LineWidth',       plt.lw.thin)
    [~, ~, p_h] = plotMean(plt_dat_x, plt_dat_y, ...
                           plt.color.o1);
    hold off
    set(p_h(1), ...
        'MarkerSize', plt.size.mrk_mean)
    set(p_h(2:4), ...
        'LineWidth', plt.lw.thick)
    axis([ax_scale ax_scale], 'square')
    if diff(ax_scale) < 200
        xticks(0:50:5000)
        yticks(0:50:5000)
    elseif diff(ax_scale) < 1000
        xticks(0:100:5000)
        yticks(0:100:5000)
    else
        xticks(0:200:5000)
        yticks(0:200:5000)
    end
    xlabel(plt_xLab)
    ylabel(plt_yLab)
    title(plt_title)
    box off

end