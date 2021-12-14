function [] = infSampling_plt_figSix(inp_dat, plt)

    % Plots latencies of first gaze shifts to different stimulus types
    % (chosen/not-chosen, easy/hard, larger/smaller set, closer/distant stimulus)
    % Input
    % inp_dat: 3D matrix with data. Rows are subjects, columns are data for
    %          x-/y-axis of plots, pages are different data types (i.e.,
    %          chosen/not-chosen, etc.)
    % plt:     structure with some general plot settings
    % Output
    % --

    %% Settings
    lab_x      = {'Latency chosen set [ms]';     'Latency easy set [ms]'; ...
                  'Latency smaller set [ms]';    'Latency closer stimulus [ms]'};
    lab_y      = {'Latency not-chosen set [ms]'; 'Latency hard set [ms]'; ...
                  'Latency larger set [ms]';     'Latency distant stimulus [ms]'};
    lab_title  = {'To chosen set'; 'To easy set'; 'To smaller set'; 'To closer stimulus'};


    %% Plot
    no_sp    = size(inp_dat, 3);
    ax_scale = [round(min(inp_dat(:)) - (min(inp_dat(:)) * 0.05)) ...
                round(max(inp_dat(:)) + (max(inp_dat(:)) * 0.05))];

    fig.h = figure;
    for sp = 1:no_sp % Subplot

        % Create plot
        subplot(2, 2, sp);
        hold on
        line([ax_scale(1) ax_scale(2)], [ax_scale(1) ax_scale(2)], ...
              'LineWidth', plt.lw.thick, ...
              'Color',     plt.color.c1, ...
              'LineStyle', '--');
        plot(inp_dat(:, 1, sp), inp_dat(:, 2, sp), ...                   Single subject data
             'o', ...
             'MarkerSize',      plt.size.mrk_ss, ...
             'MarkerFaceColor', plt.color.o2, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth',       plt.lw.thin)
        [~, ~, p_h] = plotMean(inp_dat(:, 1, sp), inp_dat(:, 2, sp), ... Mean data
                               plt.color.o1);
        set(p_h(1), ...
            'MarkerSize',      plt.size.mrk_mean, ...
            'MarkerEdgeColor', plt.color.white)
        set(p_h(2:4), ...
            'LineWidth', plt.lw.thick)
        hold off
        xlabel(lab_x{sp})
        ylabel(lab_y{sp})
        axis([ax_scale ax_scale], 'square')
        xticks(0:20:ax_scale(2))
        yticks(0:20:ax_scale(2))
        title(lab_title{sp})
        box off

    end


    %% Panel labels and write to drive
    sublabel([], -10, -25);
    opt.size    = [25 25];
    opt.imgname = strcat(plt.name.aggr(1:end-14), 'figure6');
    opt.save    = plt.save;
    prepareFigure(fig.h, opt)
    close; clear fig opt

end