function infSampling_plt_propCorrect(inp_propC_easy, inp_propC_hard, plt)

    % Plot proportion correct for easy and difficult target
    % Input
    % inp_propC_easy/_hard: vector with proportion correct of individual
    %                       subjects, seperately for easy and difficult
    %                       target
    % plt:                  structure with general plot settings

    %% Define axis limits and coordinates for illustrative lines
    ax_scale = [0.40 1];
    l_coord  = [[[ax_scale(1) ax_scale(2)]  [ax_scale(1) ax_scale(2)]]; ...
                [[ax_scale(1) ax_scale(2)], [0.5 0.5]]; ...
                [[0.5 0.5],                 [ax_scale(1) ax_scale(2)]]];


    %% Plot
    hold on
    l_h = NaN(size(l_coord, 1), 1);                            % Illustrative lines
    for l = 1:3

        l_h(l) = line(l_coord(l, 1:2), l_coord(l, 3:4), ...
                      'LineWidth', plt.lw.thick, ...
                      'Color',     plt.color.c1, ...
                      'LineStyle', '--');

    end
    plot(inp_propC_easy, inp_propC_hard, ...                   % Single subject
         'o', ...
         'MarkerSize',      plt.size.mrk_ss, ...
         'MarkerFaceColor', plt.color.o2, ...
         'MarkerEdgeColor', plt.color.white, ...
         'LineWidth',       plt.lw.thin)
    
    [~, ~, p_h] = plotMean(inp_propC_easy, inp_propC_hard, ... % Mean
                           plt.color.o1);
    set(p_h(1), ...
        'MarkerSize',      plt.size.mrk_mean, ...
        'MarkerEdgeColor', 'none')
    set(p_h(2:4), ...
        'LineWidth', plt.lw.thick)
    hold off
    axis([ax_scale ax_scale], 'square')
    xlabel('Easy target')
    ylabel('Difficult target')
    xticks(0:0.25:ax_scale(2))
    yticks(0:0.25:ax_scale(2))
    box off

end