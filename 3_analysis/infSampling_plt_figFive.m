function [] = infSampling_plt_figFive(dat_mean, dat_single, plt)

    % 
    % Input
    %
    % Output
    % --

    %% Settings
    x_lab      = {[]; []; '# gaze shift after trial start'; []};
    y_lab      = {[]; []; 'Proportion gaze shifts'; []};
    tit        = {'On chosen set'; ...
                  'On easy set'; ...
                  'On smaller set'; ...
                  'On closer stimulus'};
    x_scale    = 0.5;
    x_bound    = [0 9+x_scale];
    y_bound    = [0 1];
    x_offset   = 0.20;
    loc_chance = [0.50 0.50 0.50 1/10];


    %% Plot
    no_sp = numel(dat_mean);

    fig.h = figure;
    for sp = 1:no_sp % Subplot

        % Get data
        dat_sp       = dat_mean{sp};
        x_dat_mean   = dat_sp{1}(:, 1);               % Position of gaze shifts in trial
        y_dat_mean   = dat_sp{1}(:, 2);               % Mean proportion gaze shifts on stimulus
        ci_dat       = dat_sp{1}(:, 3);               % Confidence intervals
        y_dat_single = dat_single(:, x_dat_mean, sp); %

        % Plot
        subplot(2, 2, sp)
        hold on
        line(x_bound, [loc_chance(sp) loc_chance(sp)], ...
             'LineStyle',        '--', ...
             'Color',            plt.color.c1, ...
             'LineWidth',        plt.lw.thick, ...
             'HandleVisibility', 'off');
        plot(x_dat_mean+x_offset, y_dat_single, ...
             'o', ...
             'MarkerSize',      plt.size.mrk_ss, ...
             'MarkerFaceColor', plt.color.o2, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth',       plt.lw.thin)
        plot(x_dat_mean+x_offset, y_dat_single, ...
             '-', ...
             'Color',     plt.color.o2, ...
             'LineWidth', plt.lw.thin)
        plot(x_dat_mean, y_dat_mean, ...
             'o', ...
             'MarkerSize',      plt.size.mrk_mean, ...
             'MarkerFaceColor', plt.color.o1, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth',       plt.lw.thin)
        plot(x_dat_mean, y_dat_mean, ...
             '-', ...
             'Color',     plt.color.o1, ...
             'LineWidth', plt.lw.thick)
        errorbar(x_dat_mean, y_dat_mean, ci_dat, ...
                 'Color',            plt.color.o1, ...
                 'LineWidth',        plt.lw.thick, ...
                 'HandleVisibility', 'off')
        hold off
        axis([x_bound y_bound], 'square')
        xticks(1:1:8)
        yticks(0:0.25:1)
        xlabel(x_lab{sp})
        ylabel(y_lab{sp})
        title(tit{sp})
        box off

    end


    %% Panel labels and write to drive
    sublabel([], -10, -25);
    opt.size    = [25 25];
    opt.imgname = strcat(plt.name.aggr(1:end-14), 'figure5');
    opt.save    = plt.save;
    prepareFigure(fig.h, opt)
    close; clear fig opt

end