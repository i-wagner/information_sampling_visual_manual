function infSampling_plt_fig3(dat_searchTime, dat_searchTime_long, dat_searchTime_mod, dat_propGs, dat_searchPerf, inp_pltName, plt)

    % Plots proportion gaze shifts on AOIs, search time for different set-sizes, 
    % proportion correct, planning time inspection time and decision time 
    % for easy/difficult target
    % Input
    % dat_searchTime:      search time for individual subjects (time subjects
    %                      spent searching for targets as a function of set
    %                      size). Rows are set-sizes, columns are subjects; the
    %                      first columns is a marker variable for which
    %                      set-size group the line in the matrix belongs to
    % dat_searchTime_long: same as "dat_searchTime", but in long-format [legacy]
    % dat_searchTime_mod:  fitted regression model for search time
    % dat_propGs:          proportion gaze shifts on different AOIs;
    %                      columns are subjects, rows are different AOIs; the
    %                      first columns is a marker variable for which [legacy]
    %                      set-size group the line in the matrix belongs to
    % dat_searchPerf: data about subjects search behavior; rows are subjects,
    %                      columns are data for easy/difficult targets, 
    %                      pages are different data types 
    %                      (proportion correct, planning time, inspection time,
    %                      decision time)
    % inp_pltName:         name for output figure; defined, so we can reuse
    %                      the function easily
    % plt:                 structure with general plot settings
    % Output
    % --

    %% Proportion gaze shifts on AOI
    x_ax_offset = 0.25;
    dat_gs_mean = mean(dat_propGs(:, 2:end), 2, 'omitnan');
    dat_gs_ci   = ci_mean(dat_propGs(:, 2:end)');

    fig.h = figure;
    subplot(2, 3, 1)
    if ~sum(dat_propGs(2, 2:end), 'omitnan')

        plot(dat_propGs(1:2, 1)+x_ax_offset, dat_propGs([1, 3], 2:end), ...
              'o', ...
             'MarkerSize',      plt.size.mrk_ss, ...
             'MarkerFaceColor', plt.color.o2, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth',       plt.lw.thin, ...
             'Color',           plt.color.o2)
        hold on
        errorbar(dat_propGs(1:2, 1), dat_gs_mean([1, 3]), dat_gs_ci([1, 3]), ...
                 'o', ...
                 'MarkerSize',      plt.size.mrk_mean, ...
                 'MarkerFaceColor', plt.color.o1, ...
                 'MarkerEdgeColor', 'none', ...
                 'LineWidth',       plt.lw.thick, ...
                 'CapSize',         0, ...
                 'Color',           plt.color.o1)
        hold off
        axis([0 3 0 1], 'square')
        xticks(1:1:2)
        yticks(0:0.25:1)
        xticklabels({'Set elements', 'Background',})
        ylabel('Proportion gaze shifts on AOI')
        box off

    else

        plot(dat_propGs(:, 1)+x_ax_offset, dat_propGs(:, 2:end), ...
              'o', ...
             'MarkerSize',      plt.size.mrk_ss, ...
             'MarkerFaceColor', plt.color.o2, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth',       plt.lw.thin, ...
             'Color',           plt.color.o2)
        hold on
        errorbar(dat_propGs(:, 1), dat_gs_mean, dat_gs_ci, ...
                 'o', ...
                 'MarkerSize',      plt.size.mrk_mean, ...
                 'MarkerFaceColor', plt.color.o1, ...
                 'MarkerEdgeColor', 'none', ...
                 'LineWidth',       plt.lw.thick, ...
                 'CapSize',         0, ...
                 'Color',           plt.color.o1)
        hold off
        axis([0 4 0 1], 'square')
        xticks(1:1:3)
        yticks(0:0.25:1)
        xticklabels({'Chosen set', 'Not-chosen set', 'Background',})
        ylabel('Proportion gaze shifts on AOI')
        box off

    end


    %% Search time
    xLab_st = '# distractors [chosen set]';
    if contains(inp_pltName, 'Supp1')

        xLab_st = '# distractors [easy set]';

    end

    ax_lim       = [min(min(dat_searchTime(:, 2:end))) - 100; max(max(dat_searchTime(:, 2:end))) + 100];
    dat_reg_mean = mean(dat_searchTime(:, 2:end), 2, 'omitnan');
    dat_reg_ci   = ci_mean(dat_searchTime(:, 2:end)');
%     [lm_fit, lm_ci] = predict(mod_reg);

%     li_trials_long = dat_reg_long(:, 1) > 0;

    subplot(2, 3, 2)
    plot(dat_searchTime(:, 1)+x_ax_offset, dat_searchTime(:, 2:end), ...
         '-o', ...
         'MarkerSize',      plt.size.mrk_ss, ...
         'MarkerFaceColor', plt.color.o2, ...
         'MarkerEdgeColor', plt.color.white, ...
         'LineWidth',       plt.lw.thin, ...
         'Color',           plt.color.o2)
    hold on
%     plot(dat_reg_long(li_trials_long, 1), lm_fit(li_trials_long), ...
%          '-', ...
%          'LineWidth', plt.lw.thick, ...
%          'Color',     plt.color.black)
% %     plot(dat_reg_long(li_trials_long, 1), lm_ci(li_trials_long, :), ...
%          '--', ...
%          'LineWidth', plt.lw.thick, ...
%          'Color',     plt.color.black)
    errorbar(dat_searchTime(:, 1), dat_reg_mean, dat_reg_ci, ...
             '-o', ...
             'MarkerSize',      plt.size.mrk_mean, ...
             'MarkerFaceColor', plt.color.o1, ...
             'MarkerEdgeColor', 'none', ...
             'LineWidth',       plt.lw.thick, ...
             'CapSize',         0, ...
             'Color',           plt.color.o1)
    hold off
    axis([-1 9 ax_lim(1) ax_lim(2)], 'square')
    xticks(0:2:8)
    yticks(500:1000:10000)
    xlabel(xLab_st)
    ylabel('Search time [ms]')
    box off


    %% Proportion correct
    propC_single = dat_searchPerf(:, 1:2, 1);
    ax_scale     = [min(min(propC_single))-0.05 1];
    lCoord_x     = [ax_scale', ax_scale',   [0.50; 0.50]];
    lCoord_y     = [ax_scale', [0.50; 0.50] ax_scale'];
    propC_mean   = [mean(propC_single(:, 1), 'omitnan') mean(propC_single(:, 2), 'omitnan')];
    propC_ci     = [ci_mean(propC_single(:, 1))         ci_mean(propC_single(:, 2))];

    subplot(2, 3, 3)
    line(lCoord_x, lCoord_y, ...
         'LineWidth', plt.lw.thick, ...
         'Color',     plt.color.c1, ...
         'LineStyle', '--');
    hold on
    plot(propC_single(:, 1), propC_single(:, 2), ...
         'o', ...
         'MarkerSize',      plt.size.mrk_ss, ...
         'MarkerFaceColor', plt.color.o2, ...
         'MarkerEdgeColor', plt.color.white, ...
         'LineWidth',       plt.lw.thin)
    [~, ~, p_h] = plotMean(propC_single(:, 1), propC_single(:, 2), ...
                           plt.color.o1);
    set(p_h(1), ...
        'MarkerSize',      plt.size.mrk_mean, ...
        'MarkerEdgeColor', 'none')
    set(p_h(2:4), ...
        'LineWidth', plt.lw.thick)
    hold off
    axis([ax_scale ax_scale], 'square')
    xlabel('Proportion correct [easy target]')
    ylabel('Proportion correct [difficult target]')
    xticks(0:0.25:ax_scale(2))
    yticks(0:0.25:ax_scale(2))
    box off


    %% Planning, inspection and decision time
    xLab = {'Planning time [easy target]';      'Inspection time [easy target]';      'Response time [easy target]'};
    yLab = {'Planning time [difficult target]'; 'Inspection time [difficult target]'; 'Response time [difficult target]'};
    for tc = 2:4 % Time component

        tc_single = dat_searchPerf(:, 1:2, tc);
        tc_mean   = mean(tc_single, 1, 'omitnan');
        tc_cis    = ci_mean(tc_single);
        ax_scale  = [min(min(tc_single)) - (min(min(tc_single)) * 0.05) ...
                     max(max(tc_single)) + (max(max(tc_single)) * 0.05)];

        subplot(2, 3, tc+2)
        line([0 5000], [0 5000], ...
            'LineWidth', plt.lw.thick, ...
            'Color',     plt.color.c1, ...
            'LineStyle', '--');
        hold on
        plot(tc_single(:, 1), tc_single(:, 2), ...
            'o', ...
            'MarkerSize',      plt.size.mrk_ss, ...
            'MarkerFaceColor', plt.color.o2, ...
            'MarkerEdgeColor', plt.color.white, ...
            'LineWidth',       plt.lw.thin)
        [~, ~, p_h] = plotMean(tc_single(:, 1), tc_single(:, 2), ...
                               plt.color.o1);
        set(p_h(1), ...
            'MarkerSize', plt.size.mrk_mean)
        set(p_h(2:4), ...
            'LineWidth', plt.lw.thick)
        hold off
        axis([ax_scale ax_scale], 'square')
        if diff(ax_scale) < 200

            xticks(0:50:5000)
            yticks(0:50:5000)

        elseif diff(ax_scale) < 1000

            xticks(0:100:5000)
            yticks(0:100:5000)

        else

            xticks(0:200:5000)
            yticks(0:200:5000)

        end
        xlabel(xLab{tc-1})
        ylabel(yLab{tc-1})
        box off

    end


    %% Panel labels and save figure
    sublabel([], -10, -25);
    opt.size    = [52 35];
    opt.imgname = inp_pltName;
    opt.save    = plt.save;
    prepareFigure(fig.h, opt)
    close; clear fig opt inp_dat_var inp_dat_reg inp_dat_gs

end