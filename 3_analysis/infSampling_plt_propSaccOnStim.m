function infSampling_plt_propSaccOnStim(propGs_stimOfInterest, x_lab, y_lab, lock, plt_c, plt)

    % Plots proportion gaze shift on a given stimulus set, as a when gaze
    % shift was executed in trial
    % Input
    % propGs_stimOfInterest: cell array, containing proportion gaze shifts
    %                        as a funcion of set size. Cell array can have
    %                        more than one entry if we want to plot
    %                        proportion gaze shifts for different groups of
    %                        set sizes
    % x_/y_lab:              labels for axis
    % lock:                  x-axis timelocked to trial start (1) or last
    %                        gaze shift in trial (-1)
    % plt_c:                 Markercolor
    % plt:                   structure with some default plot settings
    % Output
    % --

    %% Settings
    x_scale = 0.5;
    y_scale = 0.05;
    y_bound = [0 1+y_scale];
    g_mrkr  = {'-o'; '-d'; '-s'};
    if sign(lock) == 1      % Timelock to trial start

        x_bound = [0 max(cellfun('size', propGs_stimOfInterest, 1))];

        x_bound(1) = x_bound(1) + x_scale * -1;
        x_bound(2) = x_bound(2) + x_scale;

    elseif sign(lock) == -1 % Timelock to last gaze shift in trial

        x_bound = [max(cellfun('size', propGs_stimOfInterest, 1))-1 0];

    end


    %% Plot
    no_groups = size(propGs_stimOfInterest, 1);
    hold on
    for g = 1:no_groups

        x_dat  = propGs_stimOfInterest{g}(:, 1); % Position of gaze shift in trial
        y_dat  = propGs_stimOfInterest{g}(:, 2); % Mean proportion gaze shifts in stimulus
        ci_dat = propGs_stimOfInterest{g}(:, 3); % Confidence interval
        plot(x_dat, y_dat, ...
             g_mrkr{g}, ...
             'MarkerSize',      plt.size.mrk_mean, ...
             'MarkerFaceColor', plt_c, ...
             'MarkerEdgeColor', plt.color.white, ...
             'Color',           plt_c, ...
             'LineWidth',       plt.lw.thin)
        errorbar(x_dat, y_dat, ...
                 ci_dat, ...
                 'Color',            plt_c, ...
                 'HandleVisibility', 'off')

    end
    hold off
    ln(1) = line([0 0], [y_bound(1) y_bound(2)], ... % Lock
                 'LineStyle', '--', ...
                 'LineWidth', plt.lw.thick, ...
                 'Color',     plt.color.c2, ...
                 'HandleVisibility', 'off');
    ln(2) = line(x_bound, [0.5 0.5], ...             % Chance
                 'LineStyle', '--', ...
                 'LineWidth', plt.lw.thick, ...
                 'Color',     plt.color.c2, ...
                 'HandleVisibility', 'off');
    uistack(ln, 'bottom')
    axis([x_bound y_bound], 'square')
    xticks(-20:1:20)
    yticks(0:0.25:1)
    xlabel(x_lab)
    ylabel(y_lab)
    box off

end