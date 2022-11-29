function infSampling_plt_distanceDist(inp_dat, plt)

    % Plots distributions of distance between fixation location and
    % closest stimulus in display
    % Input
    % inp_dat: three-column matrix, with x-/y-coordinates of gaze (1:2) and
    %          Euclidean distance between fixation location and closest
    %          stimulus (3)
    % plt:     structure with some general plot settings
    % Output
    % --

    %% Get # subjects and # conditions
    [no_sub, no_cond] = size(inp_dat, 1:2);


    %% Plot single subject distance distributions
    ax_hist = [];
    sp_tit  = {'Single-target' 'Double-target'};
    for c = 1:no_cond % Condition

        fig_h = figure; tiledlayout('flow');
        for s = 1:no_sub % Subject

            dat_sub   = inp_dat{s, c};
            mean_dist = mean(dat_sub(:, end), 'omitnan');

            ax_hist(s) = nexttile;
            hist_h = histogram(dat_sub(:, end), 30);
            line([mean_dist mean_dist], [0 max(hist_h.BinCounts)], ...
                 'LineWidth', plt.lw.thick, ...
                 'Color',     plt.color.black)
            if s == 1

                xlabel('Euclidean distance [fixation - closest stimulus]')
                ylabel('N')

            end
            title(['Subject ' num2str(s)])
            box off

        end
        linkaxes(ax_hist, 'xy')
        sgtitle(sp_tit{c})
        opt.imgname = strcat(plt.name.aggr(1:end-2), 'distanceDist_ss_e', num2str(c));
        opt.size    = [45 45];
        opt.save    = plt.save;
        prepareFigure(fig_h, opt)
        close; clear fig opt

    end

    
    %% Plot distance distributons aacross conditions
    ax_hist    = [];
    ax_surf    = [];
    idx_sp     = 1;
    surf_space = 100;
    fig_h      = figure; tiledlayout(2, 2);
    for c = 1:no_cond % Experiment

        % Histogram of distance distributions
        dat_exp   = vertcat(inp_dat{:, c});
        dat_exp   = dat_exp(all(~isnan(dat_exp), 2), :);
        mean_dist = mean(dat_exp(:, 3), 'omitnan');

        ax_hist(c) = nexttile(idx_sp);
        hist_h = histogram(dat_exp(:, 3), 70);
        line([mean_dist mean_dist], [0 max(hist_h.BinCounts)], ...
             'LineWidth', plt.lw.thick, ...
             'Color',     plt.color.black)
        xlabel('Euclidean distance [fixation - cloesest stimulus]')
        ylabel('N')
        title(sp_tit{c})
        box off

        % Surface plot of distance distributions as a function of gaze position
        ax_surf(c) = nexttile(idx_sp+2);
%         x_coord            = linspace(-15, 15, surf_space);
%         y_coord            = linspace(-15, 15, surf_space);
        x_coord            = linspace(min(dat_exp(:, 1)), max(dat_exp(:, 1)), surf_space);
        y_coord            = linspace(min(dat_exp(:, 2)), max(dat_exp(:, 2)), surf_space);
        [x_coord, y_coord] = ndgrid(x_coord, y_coord);
        interpolant        = scatteredInterpolant(dat_exp(:, 1), dat_exp(:, 2), dat_exp(:, 3));
        euc_dist           = interpolant(x_coord, y_coord);
        surf(x_coord, y_coord, euc_dist, ...
             'EdgeColor', 'none');
%         axis([-25 25 -15 15], 'square')
%         axis([-15 15 -15 15], 'square')
        xticks(-25:5:25)
        yticks(-15:5:15)
        xlabel('x-coordinates [°]')
        ylabel('y-coordinates [°]')
        view(2)

        idx_sp = idx_sp + 1;

    end
    set(ax_surf, 'Colormap', jet, 'CLim', [0 20])
    h_col = colorbar(ax_surf(end)); 
    h_col.Label.String = 'Euclidean Distance [fixation - closest stimulus]';
    linkaxes(ax_hist, 'xy')
    linkaxes(ax_surf, 'xy')
    opt.imgname = strcat(plt.name.aggr(1:end-2), 'distanceDist_all');
    opt.size    = [40 40];
    opt.save    = plt.save;
    prepareFigure(fig_h, opt)
    close; clear fig ‚opt

end