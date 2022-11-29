function infSampling_plt_figSupp2(prop_choices_easy, prop_choices_easy_fit, prop_choices_easy_pred, plt)

    % Plots target choice-behavior of individual subjects in double-target condition
    % Input
    % prop_choices_easy:      2D matrix with proportion trials, in which
    %                         subjects chose to respond on easy targets;
    %                         rows are different set-sizes, columns are
    %                         subjects
    % prop_choices_easy_fit:  linear regression fit of target
    %                         choice-behavior
    % prop_choices_easy_pred: x
    % plt:                    structure with general plotting options
    % Output
    % --

    %% Target choice behavior
    sp_idx    = 1;
    x_coord   = 0:size(prop_choices_easy, 1)-1;
    y_lim     = prop_choices_easy_fit(:, :, 2);
    y_lim     = round(max(y_lim(:)), 2);
    l_coord_x = [[x_coord(1)-1; x_coord(end)+1] [x_coord(end)/2; x_coord(end)/2]];
    l_coord_y = [[0.50;         0.50]           [0;              y_lim]];

    fig_h = figure;
    tiledlayout(5, 4, 'TileSpacing', 'tight', 'Padding', 'Compact');
    for sp = 1:size(prop_choices_easy, 2)

        if ~all(isnan(prop_choices_easy(:, sp)))

            nexttile
            line(l_coord_x, l_coord_y, ...
                 'LineStyle',        '--', ...
                 'LineWidth',        plt.lw.thick , ...
                 'Color',            plt.color.c1, ...
                 'HandleVisibility', 'off');
            hold on
            plot(x_coord, prop_choices_easy(:, sp), ...
                '-o', ...
                'MarkerSize',      plt.size.mrk_ss, ...
                'MarkerFaceColor', plt.color.o2, ...
                'MarkerEdgeColor', plt.color.white, ...
                'LineWidth',       plt.lw.thin, ...
                'Color',           plt.color.o2)
            plot(x_coord, prop_choices_easy_pred(:, sp), ...
                '-d', ...
                'MarkerSize',      plt.size.mrk_ss, ...
                'MarkerFaceColor', plt.color.p2, ...
                'MarkerEdgeColor', plt.color.white, ...
                'LineWidth',       plt.lw.thin, ...
                'Color',           plt.color.p2)
            plot(prop_choices_easy_fit(sp, :, 1), prop_choices_easy_fit(sp, :, 2), ...
                 '-', ...
                 'LineWidth', plt.lw.thick, ...
                 'Color',     plt.color.black);
            hold off
            axis([x_coord(1)-1 x_coord(end)+1 0 y_lim], 'square')
            xticks(x_coord(1):1:x_coord(end))
            xtickangle(0)
            yticks(0:0.25:1)
            xlabel('# easy distractors');
            ylabel('Proportion choices [easy target]');
            if sp_idx > 1

                xlabel([]);
                ylabel([]);

            end
            box off
            title(['Participant ' num2str(sp)]);

            sp_idx = sp_idx + 1;

        end

    end
    h_leg = legend({'Empirical', 'Model', 'Regression'});
    h_leg.Layout.Tile = 20;
    legend box off


    %% Panel labels and save
    opt.size    = [30 55];
    opt.imgname = strcat(plt.name.aggr(1:end-14), 'figureSupp2');
    opt.save    = plt.save;
    prepareFigure(fig_h, opt)
    close all

end