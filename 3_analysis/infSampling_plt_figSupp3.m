function infSampling_plt_figSupp3(propFix_pred, propFix_emp, plt)

    %
    %
    %
    %
    % --

    %% Init
    no_ss = size(propFix_pred, 2);


    %% Proportion fixations on chosen/easy set as a function of set size
    % Seperate for each set size
    fig_h = figure;
    for ss = 1:no_ss % Set size

        r      = corrcoef(propFix_emp(:, ss), propFix_pred(:, ss), 'Rows', 'complete');
        r_sqrt = round(r(1, 2)^2, 2);

        nexttile(ss);
        line([0 1], [0 1], ...
             'LineStyle', '--', ...
             'Color',     plt.color.c1, ...
             'LineWidth', plt.lw.thick);
        hold on
        plot(propFix_emp(:, ss), propFix_pred(:, ss), ...
             'o', ...
             'MarkerSize',      plt.size.mrk_ss, ...
             'MarkerFaceColor', plt.color.mid, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth',       plt.lw.thin)
        plot(mean(propFix_emp(:, ss), 'omitnan'), mean(propFix_pred(:, ss), 'omitnan'), ...
             'o', ...
             'MarkerSize',      plt.size.mrk_mean, ...
             'MarkerFaceColor', plt.color.dark, ...
             'MarkerEdgeColor', 'none')
        [~, ~, p_h] = plotMean(propFix_emp(:, ss), propFix_pred(:, ss), plt.color.dark);
        set(p_h(1), ...
            'MarkerSize',      plt.size.mrk_mean, ...
            'MarkerEdgeColor', 'none')
        set(p_h(2:4), ...
            'LineWidth', plt.lw.thick)
        hold off
        axis([0 1 0 1], 'square');
        xticks(0:0.25:1);
        yticks(0:0.25:1);
        if ss == 1

            xlabel('Proportion fixations chosen set [empirical]')
            ylabel('Proportion fixations chosen set [predicted]')

        end
        set(gca, 'XColor', plt.color.o1);
        set(gca, 'YColor', plt.color.p1);
        text(0.65, 0.10, ['R^2 = ' num2str(r_sqrt)]);
        if ss == 2

            title([num2str(ss)-1, ' easy distractor']);

        else

            title([num2str(ss)-1, ' easy distractors']);

        end
        box off

    end
    opt.size    = [35 35];
    opt.imgname = strcat(plt.name.aggr(1:end-14), 'figureSupp3');
    opt.save    = plt.save;
    prepareFigure(fig_h, opt)
    close

end