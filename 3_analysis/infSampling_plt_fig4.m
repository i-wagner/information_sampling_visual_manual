function infSampling_plt_fig4(prop_choices_easy, prop_choices_easy_fit, slopesIntercepts, plt)

    % Plots proportion choices easy target for three representative
    % participants from the double-target condition as well as
    % slopes/intercepts of regressions, fitted to decision-curves of
    % participants from the double-target condition 
    % Input
    % prop_choices_easy:     proportion choices easy target as a function of
    %                        set-size; columns are subjects, rows are
    %                        different set-size groups
    % prop_choices_easy_fit: fits of regressions to proportion choices easy
    %                        target as a function of set size; first page
    %                        is x-, second page is y-coordinates of fitted
    %                        regression lines. Rows are subjects, columns
    %                        are different set-size groups
    % slopesIntercepts:      intercepts (:, 1) and slopes(:, 2) of
    %                        regression fits to decision curves of
    %                        participants from the double-target condition;
    %                        rows are different participants
    % plt:                   structure with general plot settings
    % Output
    % --

    %% Data of representative participants
    % Subject 3 (trade-off set-size/discrimination difficults),
    % Subject 13 (no strategy),
    % Subject 16 (always easy)
    sub_no    = [16 3 13];
    y_lim     = prop_choices_easy_fit(:, :, 2);
    y_lim     = round(max(y_lim(:)), 2);
    x_coord   = 0:size(prop_choices_easy, 1)-1;
    l_coord_x = [[x_coord(1)-1; x_coord(end)+1] [x_coord(end)/2; x_coord(end)/2]];
    l_coord_y = [[0.50;         0.50]           [0;              y_lim]];

    fig_h = figure;
    for sp = 1:3 % Subplot

        curr_sub = sub_no(sp);

        subplot(2, 2, sp)
        line(l_coord_x, l_coord_y, ...
             'LineStyle', '--', ...
             'LineWidth', plt.lw.thick , ...
             'Color',     plt.color.c1);
        hold on
        plot(x_coord, prop_choices_easy(:, curr_sub), ...
            '-o', ...
            'MarkerSize',      plt.size.mrk_ss, ...
            'MarkerFaceColor', plt.color.o2, ...
            'MarkerEdgeColor', plt.color.white, ...
            'LineWidth',       plt.lw.thin, ...
            'Color',           plt.color.o2)
        plot(prop_choices_easy_fit(curr_sub, :, 1), prop_choices_easy_fit(curr_sub, :, 2), ...
             '-', ...
             'LineWidth', plt.lw.thick, ...
             'Color',     plt.color.black);
        hold off
        axis([x_coord(1)-1 x_coord(end)+1 0 y_lim], 'square')
        xticks(x_coord(1):1:x_coord(end))
        yticks(0:0.25:1)
        xlabel('# easy distractors');
        ylabel('Proportion choices [easy target]');
        box off
        title(['Participant ' num2str(curr_sub)]);

    end


    %% Slopes and intercepts of regression fits
    regress_mean = mean(slopesIntercepts(:, 1:2), 1, 'omitnan');
    regress_ci   = ci_mean(slopesIntercepts);
    l_coord_x    = [[-0.60; 0.60] [0;     0]];
    l_coord_y    = [[0;     0]    [-0.20; 0.20]];

    subplot(2, 2, 4);
    hold on
    line(l_coord_x, l_coord_y, ...
         'LineWidth', plt.lw.thick, ...
         'Color',     plt.color.c1, ...
         'LineStyle', '--');
    plot(slopesIntercepts(:, 1), slopesIntercepts(:, 2), ...
         'o', ...
         'MarkerSize',      plt.size.mrk_ss, ...
         'MarkerFaceColor', plt.color.o2, ...
         'MarkerEdgeColor', plt.color.white, ...
         'LineWidth',       plt.lw.thin);
    errorbar(regress_mean(1), regress_mean(2), regress_ci(2), regress_ci(2), regress_ci(1), regress_ci(1), ...
             '-o', ...
             'MarkerSize',      plt.size.mrk_mean, ...
             'MarkerFaceColor', plt.color.o1, ...
             'MarkerEdgeColor', 'none', ...
             'LineWidth',       plt.lw.thick, ...
             'CapSize',         0, ...
             'Color',           plt.color.o1);
    hold off
    xlim([-0.30 0.60]);
    ylim([-0.20 0.10]);
    axis square;
    box off
    xlabel('Intercept [difficulty]');
    ylabel('Slope [set size]');    


    %% Panel labels and save
    sublabel([], -10, -25);
    opt.size    = [25 25];
    opt.imgname = strcat(plt.name.aggr(1:end-14), 'figure4');
    opt.save    = plt.save;
    prepareFigure(fig_h, opt)
    close all

end