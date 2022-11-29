function [] = infSampling_plt_fig5(dat_single, plt)

    % Plots proportion gaze shifts to different AOIs of interest
    % Input
    % dat_single: proportion gaze shifts to AOIs of interest; rows are
    %             subjects, columns are different gaze shifts in a
    %             sequence, pages are different stimuli of interest
    % plt:        structure with general plot settings
    % Output
    % --

    %% Proportion gaze shifts to different AOIs of interest
    no_sp      = size(dat_single, 3);
    x_lab      = {[]; []; '# gaze shift after trial start'; []};
    y_lab      = {[]; []; 'Proportion gaze shifts'; []};
    tit        = {'On chosen set'; ...
                  'On easy set'; ...
                  'On smaller set'; ...
                  'On closer element'};
    x_scale    = 0.50;
    x_bound    = [1-x_scale 2+x_scale];
    y_bound    = [0 1];
    x_offset   = 0.05;
    loc_chance = [0.50 0.50 0.25 1/10];

    fig.h = figure;
    for sp = 1:no_sp % Subplot

        dat_ss   = dat_single(:, 1:2, sp);
        dat_mean = mean(dat_ss, 1, 'omitnan');
        dat_cis  = ci_mean(dat_ss);

        subplot(2, 2, sp)
        bound_chanceLine = x_bound;
        if sp == no_sp

%             bound_chanceLine(2) = 1 + x_offset;
%             line(bound_chanceLine, [loc_chance(sp) loc_chance(sp)], ...
%                  'LineStyle',        '--', ...
%                  'Color',            plt.color.c1, ...
%                  'LineWidth',        plt.lw.thick, ...
%                  'HandleVisibility', 'off');
% 
%             bound_chanceLine(1) = bound_chanceLine(2);
%             bound_chanceLine(2) = x_dat_mean(2) + x_offset;

        end
        hold on
        line(bound_chanceLine, [loc_chance(sp) loc_chance(sp)], ...
             'LineStyle',        '--', ...
             'Color',            plt.color.c1, ...
             'LineWidth',        plt.lw.thick, ...
             'HandleVisibility', 'off');
        plot((1:2)+x_offset, dat_ss, ...
             '-o', ...
             'MarkerSize',      plt.size.mrk_ss, ...
             'MarkerFaceColor', plt.color.o2, ...
             'MarkerEdgeColor', plt.color.white, ...
             'LineWidth',       plt.lw.thin, ...
             'Color',           plt.color.o2)
        errorbar((1:2), dat_mean, dat_cis, ...
                 '-o', ...
                 'MarkerSize',      plt.size.mrk_mean, ...
                 'MarkerFaceColor', plt.color.o1, ...
                 'MarkerEdgeColor', 'none', ...
                 'Color',            plt.color.o1, ...
                 'LineWidth',        plt.lw.thick, ...
                 'Capsize',          0, ...
                 'HandleVisibility', 'off')
        hold off
        axis([x_bound y_bound], 'square')
        xticks(1:1:2)
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