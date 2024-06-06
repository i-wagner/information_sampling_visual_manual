function [] = infSampling_plt_figPerf(gain_emp, gain_mod_comb_perfect, gain_mod_comb_noise, choice_emp, choice_mod_comb, plt)

    % Plots empirical gain vs gain, predicted by a model with and without
    % noise as well as empirical proportion choices for easy targets and
    % proportion choices easy target predicted by a model with noise
    % Input
    % gain_emp:              empirical gain in double-target condition;
    %                        rows are subjects
    % gain_mod_comb_perfect: gain in double-target condition, predicted by
    %                        a model without noise; rows are subjects
    % gain_mod_comb_noise:   gain in double-target condition, predicted by
    %                        a model with noise; rows are subjects
    % choice_emp:            empirical proportion choices easy targets in
    %                        double-target condition; rows are subjects,
    %                        columns are set-sizes
    % choice_mod_comb:       proportion choices easy targets in
    %                        double-target condition, predicted by a model 
    %                        with noise; rows are subjects, columns are set-sizes
    % plt:                   structure with general plot settings
    % Output
    % --

    %% Empirical vs. predicted gain
    % Empirical gain vs. gain, predicted by model with decision noise
    % Empirical gain vs. gain, predicted by model without decision noise
    dat_mod      = [gain_mod_comb_perfect gain_mod_comb_noise]; 
    tit          = {'Model without noise'; 'Model with noise'};
    ax_lim_upper = round(max([gain_emp; dat_mod(:)]), 2) + 0.10;

    fig.h = figure;
    for d = 1:2 % Data

        r         = corrcoef(gain_emp, dat_mod(:, d), 'Rows', 'complete');
        r_sqrt    = round(r(1, 2)^2, 2);
        gain_mean = [mean(gain_emp, 'omitnan') mean(dat_mod(:, d), 'omitnan')];
        gain_cis  = [ci_mean(gain_emp)         ci_mean(dat_mod(:, d))];

        subplot(1, 3, d);
        line([0 ax_lim_upper], [0 ax_lim_upper], ...
             'LineStyle', '--', ...
             'Color',     plt.color.c1, ...
             'LineWidth', plt.lw.thick)
        hold on
        plot(gain_emp, dat_mod(:, d), ...
             'o', ...
             'MarkerSize',      plt.size.mrk_ss, ...
             'MarkerFaceColor', plt.color.mid, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth',       plt.lw.thin)
        plot(gain_mean(1), gain_mean(2), ...
             'o', ...
             'MarkerSize',      plt.size.mrk_mean, ...
             'MarkerFaceColor', plt.color.dark, ...
             'MarkerEdgeColor', 'none')
        [~, ~, p_h] = plotMean(gain_emp, dat_mod(:, d), plt.color.dark);
        set(p_h(1), ...
            'MarkerSize',      plt.size.mrk_mean, ...
            'MarkerEdgeColor', 'none')
        set(p_h(2:4), ...
            'LineWidth', plt.lw.thick)
        hold off
        axis([0 ax_lim_upper 0 ax_lim_upper], 'square')
        set(gca, 'XColor', plt.color.o1);
        set(gca, 'YColor', plt.color.p1);
        xlabel('Empirical gain [Cent/s]')
        ylabel('Predicted gain [Cent/s]')
        text(0.10, 1.40, ['R^2 = ' num2str(r_sqrt)]);
        title(tit{d})
        box off

    end


    %% Empirical proportion choices easy target vs. proportion choices easy target, predicted by model with noise
    dat         = cat(3, choice_emp, choice_mod_comb);
    col         = {plt.color.o1; plt.color.p1};
    mrk         = {'o' 'd'};
    ax_x_offset = [0 0.20];

    subplot(1, 3, 3)
    line([-1 9], [0.50 0.50], ...
         'LineStyle',        '--', ...
         'Color',            plt.color.c1, ...
         'LineWidth',        plt.lw.thick, ...
         'HandleVisibility', 'off')
    hold on
    for d = 1:2 % Data

        choices_mean = mean(dat(:, :, d), 1, 'omitnan');
        choices_cis  = ci_mean(dat(:, :, d));

        plot((0:8)+ax_x_offset(d), choices_mean, ...
             mrk{d}, ...
             'MarkerSize',      plt.size.mrk_mean, ...
             'MarkerFaceColor', col{d, :}, ...
             'MarkerEdgeColor', 'none')
        plot((0:8)+ax_x_offset(d), choices_mean, ...
             '-', ...
             'Color',            col{d, :}, ...
             'LineWidth',        plt.lw.thick, ...
             'HandleVisibility', 'off')
        errorbar((0:8)+ax_x_offset(d), choices_mean, choices_cis, ...
                 'Color',            col{d, :}, ...
                 'LineWidth',        plt.lw.thick, ...
                 'Capsize',          0, ...
                 'HandleVisibility', 'off')

    end
    hold off
    axis([-1 9 0 1], 'square')
    xticks(0:2:8)
    yticks(0:0.25:1)
    xlabel('# easy distractors')
    ylabel('Proportion choices [easy target]')
    box off
    legend('Empirical', 'Model with noise', 'Location', 'SouthWest');
    legend box off


    %% Plot panel labels and save figure
    sublabel([], -10, -25);
    opt.size    = [45 15];
    opt.imgname = strcat('performance');
    opt.save    = 1;
    prepareFigure(fig.h, opt)
    close

end