function infSampling_plt_figTwo(dat_reg, dat_reg_long, mod_reg, dat_gs, dat_var, plt)

    % Plots some manipulation checks (proportion gaze shifts on AOIs, search 
    % time for different set-sizes, proportion correct, planning time
    % inspexction time and decision time for easy/hard target)
    % Input
    % dat_reg: data for regression either in long or short format; in short
    %          format columns are subjects, rows are set sizes
    % mod_reg: fitted regression model for search time
    % dat_gs:  proportion gaze shifts on different AOIs; columns are
    %          subjects, rows are different AOIs
    % dat_var: data about subjects serch behavior; rows are subjects,
    %          columns are data from easy/hard target, pages are different
    %          data types (proportion correct, etc.)
    % Output
    % --

    %% Plot proportion gaze shifts on AOI
    x_ax_offset = 0.25;

    fig.h = figure;
    subplot(2, 3, 1)
    plot(dat_gs(:, 1)+x_ax_offset, dat_gs(:, 2:end), ...
          '-o', ...
         'MarkerSize',      plt.size.mrk_ss, ...
         'MarkerFaceColor', plt.color.o2, ...
         'MarkerEdgeColor', plt.color.white, ...
         'LineWidth',       plt.lw.thin, ...
         'Color',           plt.color.o2)
    hold on
    errorbar(dat_gs(:, 1), mean(dat_gs(:, 2:end), 2, 'omitnan'), ...
             ci_mean(dat_gs(:, 2:end)'), ...
             'LineWidth', plt.lw.thick, ...
             'Color',     plt.color.o1)
    plot(dat_gs(:, 1), mean(dat_gs(:, 2:end), 2, 'omitnan'), ...
         'o', ...
         'MarkerSize',      plt.size.mrk_mean, ...
         'MarkerFaceColor', plt.color.o1, ...
         'MarkerEdgeColor', 'None', ...
         'LineWidth',       plt.lw.thin)
    hold off
    axis([0 6 0 0.5], 'square')
    xticks(1:1:5)
    yticks(0:0.25:1)
    xticklabels({'Easy target', 'Difficult target', 'Easy distractors', 'Difficult distractors', 'Background',})
    ylabel('Proportion gaze shifts in AOI')
    box off


    %% Plot regression fit
    [lm_fit, lm_ci] = predict(mod_reg);

    li_trials      = sum(dat_reg > 0,2 ) > 0;
    li_trials_long = dat_reg_long(:, 1) > 0;

    subplot(2, 3, 2)
    plot(dat_reg(li_trials, 1)+x_ax_offset, dat_reg(li_trials, 2:end), ...
         '-o', ...
         'MarkerSize',      plt.size.mrk_ss, ...
         'MarkerFaceColor', plt.color.o2, ...
         'MarkerEdgeColor', plt.color.white, ...
         'LineWidth',       plt.lw.thin, ...
         'Color',           plt.color.o2)
    hold on
    plot(dat_reg_long(li_trials_long, 1), lm_fit(li_trials_long), ...
         '-', ...
         'LineWidth', plt.lw.thick, ...
         'Color',     plt.color.black)
    plot(dat_reg_long(li_trials_long, 1), lm_ci(li_trials_long, :), ...
         '--', ...
         'LineWidth', plt.lw.thick, ...
         'Color',     plt.color.black)
    errorbar(dat_reg(li_trials, 1), mean(dat_reg(li_trials, 2:end), 2, 'omitnan'), ci_mean(dat_reg(li_trials, 2:end)'), ...
             '.', ...
             'LineWidth',       plt.lw.thick, ...
             'Color',           plt.color.o1)
    plot(dat_reg(li_trials, 1), mean(dat_reg(li_trials, 2:end), 2, 'omitnan'), ...
         'o', ...
         'MarkerSize',      plt.size.mrk_mean, ...
         'MarkerFaceColor', plt.color.o1, ...
         'MarkerEdgeColor', plt.color.white, ...
         'LineWidth',       plt.lw.thin)
    hold off
    axis([-1 9 200 4500], 'square')
    xticks(0:2:8)
    yticks(500:1000:4500)
    xlabel('# distractors')
    ylabel('Search time [ms]')
    box off
    clear li_trials


    %% Plot proportion correct and search parameter
    var_xLab   = {[], 'Easy target [ms]', 'Easy target [ms]', 'Easy target [ms]'};
    var_yLab   = {[], 'Difficult target [ms]', 'Difficult target [ms]', 'Difficult target [ms]'};
    var_title  = {'Proportion correct'; 'Planning time'; ...
                  'Inspection time'; 'Decision time'};
    for sp = 1:4 % Subplot

        subplot(2, 3, sp+2)
        if sp == 1 % Proportion correct

            infSampling_plt_propCorrect(dat_var(:, 1, sp), dat_var(:, 2, sp), plt);
            title(var_title{sp})

        else       % Planning-/search-/decision-time

            infSampling_plt_time(dat_var(:, 1, sp), dat_var(:, 2, sp), ...
                                 var_xLab{sp}, var_yLab{sp}, ...
                                 var_title{sp}, plt)

        end

    end


    %% Plot panel labels and save figure
    sublabel([], -10, -25);
    opt.size    = [45 30];
    opt.imgname = strcat(plt.name.aggr(1:end-14), 'figure2');
    opt.save    = plt.save;
    prepareFigure(fig.h, opt)
    close; clear fig opt inp_dat_var inp_dat_reg inp_dat_gs

end