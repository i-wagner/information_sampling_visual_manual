function [] = infSampling_plt_fig6(inp_emp_propChoicesEasy, inp_emp_propGsChosen, inp_emp_perf, ...
                                   model, all, plt)

    % Plot results of model fitting
    % WHERE APPLICABLE: SUBJECTS ARE ROWS, COLUMNS ARE SET-SIZES
    %
    % Input
    % inp_emp_propChoicesEasy: set-size wise proportion choice for easy
    %                          targets
    % inp_emp_propGsChosen:    set-size wise proportion gaze shifts on
    % inp_emp_perf:            average empirical performance
    % model:                   structure with fitting results of stochastic
    %                          model
    % all:                     structure with fitting results of noise
    %                          model
    % plt:                     structure with general plot settings
    % 
    % Output
    % --

    %% Empirical vs. perfect gain
    gain_perfect = all.perf_perfect(:, 3);
    gain_model   = model.performance(:, 2);         % Complex model
    dat_gain     = [gain_perfect gain_model]; 

    fig.h = figure;
    ax_lim_upper   = round(max([inp_emp_perf; dat_gain(:)]), 2) + 0.10;
    sp_offset      = [0 1];
    axCol_x        = {plt.color.black, plt.color.o1};
    axCol_y        = {plt.color.black, plt.color.p1};
    axLab_y        = {'Optimal gain [Cent/s]', 'Predicted gain [Cent/s]'};
    markerCol_mean = {plt.color.o1 plt.color.dark};
    markerCol_ss   = {plt.color.o2 plt.color.mid};
    for d = 1:1 % Data

        r         = corrcoef(inp_emp_perf, dat_gain(:, d), 'Rows', 'complete');
        r_sqrt    = round(r(1, 2)^2, 2);
        gain_mean = [mean(inp_emp_perf, 'omitnan') mean(dat_gain(:, d), 'omitnan')];

        subplot(2, 3, d+sp_offset(d));
        line([0 ax_lim_upper], [0 ax_lim_upper], ...
             'LineStyle', '--', ...
             'Color',     plt.color.c1, ...
             'LineWidth', plt.lw.thick)
        hold on
        plot(inp_emp_perf, dat_gain(:, d), ...
             'o', ...
             'MarkerSize',      plt.size.mrk_ss, ...
             'MarkerFaceColor', markerCol_ss{d}, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth',       plt.lw.thin)
        plot(gain_mean(1), gain_mean(2), ...
             'o', ...
             'MarkerSize',      plt.size.mrk_mean, ...
             'MarkerFaceColor', markerCol_mean{d}, ...
             'MarkerEdgeColor', 'none')
        [~, ~, p_h] = plotMean(inp_emp_perf, dat_gain(:, d), markerCol_mean{d});
        set(p_h(1), ...
            'MarkerSize',      plt.size.mrk_mean, ...
            'MarkerEdgeColor', 'none')
        set(p_h(2:4), ...
            'LineWidth', plt.lw.thick)
        hold off
        axis([0 ax_lim_upper 0 ax_lim_upper], 'square')
        set(gca, 'XColor', axCol_x{d});
        set(gca, 'YColor', axCol_y{d});
        xlabel('Empirical gain [Cent/s]')
        ylabel(axLab_y{d})
        text(0.10, 1.40, ['R^2 = ' num2str(r_sqrt)]);
        box off

    end


    %% Model parameters
    x_jit = 0.075 .* randn(size(model.freeParameter{2}, 1), 1);
    y_dat = fliplr(model.freeParameter{2});

    subplot(2, 3, 2)
    plot([1+x_jit 2+x_jit], y_dat, ...
         'o', ...
         'MarkerSize',      plt.size.mrk_ss, ...
         'MarkerFaceColor', plt.color.o2, ...
         'MarkerEdgeColor', plt.color.white, ...
         'LineWidth',       plt.lw.thin)
    hold on
    errorbar((1:2)+0.25, mean(y_dat, 1, 'omitnan'), ci_mean(y_dat), ...
             'o', ....
             'MarkerSize',       plt.size.mrk_mean, ...
             'MarkerFaceColor',  plt.color.o1, ...
             'MarkerEdgeColor',  plt.color.white, ...
             'LineWidth',        plt.lw.thick, ...
             'Color',            plt.color.o1, ...
             'Capsize',          0, ...
             'HandleVisibility', 'off')
    hold off
    axis([0.50 2.50 0 ceil(max(y_dat(:))) + 0.10], 'square')
    xticks(1:1:2)
    yticks(0:0.50:2)
    xlabel('Noise parameter')
    xticklabels({'Decision'; 'Fixation'})
    ylabel('Parameter value')
    box off


    %% Empirical vs. stochastic model gain
    gain_perfect = all.perf_perfect(:, 3);
    gain_model   = model.performance(:, 2);         % Complex model
    dat_gain     = [gain_perfect gain_model]; 

    ax_lim_upper   = round(max([inp_emp_perf; dat_gain(:)]), 2) + 0.10;
    sp_offset      = [0 1];
    axCol_x        = {plt.color.black, plt.color.o1};
    axCol_y        = {plt.color.black, plt.color.p1};
    axLab_y        = {'Optimal gain [Cent/s]', 'Predicted gain [Cent/s]'};
    markerCol_mean = {plt.color.o1 plt.color.dark};
    markerCol_ss   = {plt.color.o2 plt.color.mid};
    for d = 2:2 % Data

        r         = corrcoef(inp_emp_perf, dat_gain(:, d), 'Rows', 'complete');
        r_sqrt    = round(r(1, 2)^2, 2);
        gain_mean = [mean(inp_emp_perf, 'omitnan') mean(dat_gain(:, d), 'omitnan')];

        subplot(2, 3, d+sp_offset(d));
        line([0 ax_lim_upper], [0 ax_lim_upper], ...
             'LineStyle', '--', ...
             'Color',     plt.color.c1, ...
             'LineWidth', plt.lw.thick)
        hold on
        plot(inp_emp_perf, dat_gain(:, d), ...
             'o', ...
             'MarkerSize',      plt.size.mrk_ss, ...
             'MarkerFaceColor', markerCol_ss{d}, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth',       plt.lw.thin)
        plot(gain_mean(1), gain_mean(2), ...
             'o', ...
             'MarkerSize',      plt.size.mrk_mean, ...
             'MarkerFaceColor', markerCol_mean{d}, ...
             'MarkerEdgeColor', 'none')
        [~, ~, p_h] = plotMean(inp_emp_perf, dat_gain(:, d), markerCol_mean{d});
        set(p_h(1), ...
            'MarkerSize',      plt.size.mrk_mean, ...
            'MarkerEdgeColor', 'none')
        set(p_h(2:4), ...
            'LineWidth', plt.lw.thick)
        hold off
        axis([0 ax_lim_upper 0 ax_lim_upper], 'square')
        set(gca, 'XColor', axCol_x{d});
        set(gca, 'YColor', axCol_y{d});
        xlabel('Empirical gain [Cent/s]')
        ylabel(axLab_y{d})
        text(0.10, 1.40, ['R^2 = ' num2str(r_sqrt)]);
        box off

    end


    %% Empirical vs. predicted proportion choices easy target
    dat         = cat(3, inp_emp_propChoicesEasy, model.propChoicesEasy(:, :, 2));
    col         = {plt.color.o1; plt.color.p1};
    mrk         = {'o' 'd'};
    ax_x_offset = [0 0.20];

    subplot(2, 3, [4 5])
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
    axis([-1 9 0 1])
    xticks(0:2:8)
    yticks(0:0.25:1)
    xlabel('# easy distractors')
    ylabel('Proportion choices [easy target]')
    box off
    legend('Empirical', 'Model', 'Location', 'SouthWest');
    legend box off


    %% Overall proportions fixations on chosen/easy set
    propFix_pred_mean = mean(model.propFixChosen(:, :, 2), 2);
    propFix_emp_mean  = mean(inp_emp_propGsChosen, 2);
    ax_lim            = round(min([propFix_emp_mean; propFix_pred_mean])-0.10, 2);
    rSquared          = corrcoef(propFix_pred_mean, propFix_emp_mean, 'Rows', 'Complete');
    rSquared          = round(rSquared(1, 2).^2, 2);

    subplot(2, 3, 6)
    line([0 1], [0 1], ...
         'LineStyle', '--', ...
         'Color',     plt.color.c1, ...
         'LineWidth', plt.lw.thick)
    hold on
    plot(propFix_emp_mean, propFix_pred_mean, ...
         'o', ...
         'MarkerSize',      plt.size.mrk_ss, ...
         'MarkerFaceColor', plt.color.mid, ...
         'MarkerEdgeColor', plt.color.white, ...
         'LineWidth',       plt.lw.thin)
    plot(mean(propFix_emp_mean, 'omitnan'), mean(propFix_pred_mean, 'omitnan'), ...
         'o', ...
         'MarkerSize',      plt.size.mrk_mean, ...
         'MarkerFaceColor', markerCol_mean{d}, ...
         'MarkerEdgeColor', 'none')
    [~, ~, p_h] = plotMean(propFix_emp_mean, propFix_pred_mean, plt.color.dark);
    set(p_h(1), ...
        'MarkerSize',      plt.size.mrk_mean, ...
        'MarkerEdgeColor', 'none')
    set(p_h(2:4), ...
        'LineWidth', plt.lw.thick)
    hold off
    text(0.53, 0.96, ['R^2 = ' num2str(rSquared)]);
    axis([ax_lim 1 ax_lim 1], 'square')
    xticks(0:0.25:1)
    yticks(0:0.25:1)
    xlabel('Empirical fixations chosen set [proportions]')
    ylabel('Predicted fixations chosen set [proportions]')
    set(gca, 'XColor', plt.color.o1);
    set(gca, 'YColor', plt.color.p1);


    %% Panel labels and write to drive
    sublabel([], -10, -25);
    opt.size    = [45 30];
    opt.imgname = strcat(plt.name.aggr(1:end-14), 'figure6');
    opt.save    = plt.save;
    prepareFigure(fig.h, opt)
    close; clear fig opt

end