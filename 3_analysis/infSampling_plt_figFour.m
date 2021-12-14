function [] = infSampling_plt_figFour(gain_emp, gain_mod_comb_perfect, gain_mod_comb_noise, choice_emp, choice_mod_comb, plt)

    % 
    % Input
    %
    % Output
    % --

    %% Empirical vs. predicted gain
    % Empirical gain vs. gain, predicted by model with decision noise
    % Empirical gain vs. gain, predicted by model without decision noise
    dat_mod      = [gain_mod_comb_perfect           gain_mod_comb_noise]; 
    tit          = {'Model without decision noise'; 'Model with decision noise'};
    ax_lim_upper = round(max([gain_emp; dat_mod(:)]), 2) + 0.10;

    fig.h = figure;
    for d = 1:2 % Data

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
%         plotMean(gain_emp, dat_mod(:, d), plt.color.mid);
        hold off
        xlabel('Empirical gain [cent/s]')
        ylabel('Predicted gain [cent/s]')
        title(tit{d})
        axis([0 ax_lim_upper 0 ax_lim_upper], 'square')
        set(gca, 'XColor', plt.color.o1);
        set(gca, 'YColor', plt.color.p1);
        box off

    end


    %% Empirical proportion choices easy target vs. proportion choices easy target, predicted by model with decision noise
    dat         = cat(3, choice_emp, choice_mod_comb);
    col         = {plt.color.o1; plt.color.p1};
    mrk         = {'o' 'd'};
    ax_x_offset = [0 0.20];
    subplot(1, 3, 3)
    line([-1 9], [0.5 0.5], ...
         'LineStyle',        '--', ...
         'Color',            plt.color.c1, ...
         'LineWidth',        plt.lw.thick, ...
         'HandleVisibility', 'off')
    hold on
    for d = 1:2 % Data

        ci = ci_mean(dat(:, :, d));
        plot((0:8)+ax_x_offset(d), mean(dat(:, :, d), 'omitnan'), ...
             mrk{d}, ...
             'MarkerSize',      plt.size.mrk_mean, ...
             'MarkerEdgeColor', plt.color.white, ...
             'MarkerFaceColor', col{d, :}, ...
             'LineWidth',       plt.lw.thin)
        plot((0:8)+ax_x_offset(d), mean(dat(:, :, d), 'omitnan'), ...
             '-', ...
             'Color',            col{d, :}, ...
             'LineWidth',        plt.lw.thick, ...
             'HandleVisibility', 'off')
        errorbar((0:8)+ax_x_offset(d), mean(dat(:, :, d), 'omitnan'), ci, ...
                 'Color',     col{d, :}, ...
                 'LineWidth', plt.lw.thick, ...
                 'HandleVisibility', 'off')

    end
    hold off
    axis([-1 9 0 1], 'square')
    xticks(0:2:8)
    yticks(0:0.25:1)
    xlabel('# easy distractors')
    ylabel('Proportion choices easy target')
    box off
    legend('Empirical', 'Model with noise', 'Location', 'SouthWest');
    legend box off


    %% Plot panel labels and save figure
    sublabel([], -10, -25);
    opt.size    = [45 15];
    opt.imgname = strcat(plt.name.aggr(1:end-14), 'figure4');
    opt.save    = plt.save;
    prepareFigure(fig.h, opt)
    close; clear fig opt inp_dat_var inp_dat_reg inp_dat_gs

end