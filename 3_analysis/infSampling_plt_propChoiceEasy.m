function infSampling_plt_propChoiceEasy(propChoiceEasy, plt, do_mean)

    % Plots proportion choices easy target as a function of # easy distractors
    % Input
    % propChoiceEasy: vector containing proportion choices for easy target
    %                 as a function of # easy distractors; data has to be
    %                 ordered, so the first entry corresponds to proportion
    %                 choices for 0 easy distractors and the last entry
    %                 corresponds to proportion choices for 8 easy
    %                 distractors. Can be a matrix if we want to plot
    %                 aggregated data; rows have to be proportion choices
    %                 for different set-sizes and columns ahve to subjects
    % plt:            structure with general plot-related variables
    % do_mean:        flag, indicating if we plot the mean over all
    %                 subjects (1) or not (0)
    % Output
    % --

    %% Check input and assign default variable values
    if nargin < 3

        do_mean = 0;

    end


    %% Calculate mean and confidence intervals
    if do_mean == 1

        ci_propChoiceEasy = ci_mean(propChoiceEasy')';
        propChoiceEasy    = mean(propChoiceEasy, 2, 'omitnan');

    end


    %% Plot data
    x_coord = 0:numel(propChoiceEasy)-1;

    l_h(1) = line([-1 9], [0.5 0.5], ...
                  'LineStyle', '--', ...
                  'LineWidth', plt.lw.thick , ...
                  'Color',     plt.color.c1);
    l_h(2) = line([4 4], [0 1], ...
                  'LineStyle', '--', ...
                  'LineWidth', plt.lw.thick , ...
                  'Color',     plt.color.c1);
    hold on
    plot(x_coord, propChoiceEasy, ...
        '-o', ...
        'MarkerSize',      plt.size.mrk_ss, ...
        'MarkerFaceColor', plt.color.o2, ...
        'MarkerEdgeColor', plt.color.white, ...
        'LineWidth',       plt.lw.thick, ...
        'Color',           plt.color.o2)
    if do_mean == 1 % Plot aggregated data

        errorbar(0:8, propChoiceEasy, ...
                 ci_propChoiceEasy, ...
                 'LineWidth', plt.lw, ...
                 'Color',     plt.color.o1)

    end
    hold off
    axis([x_coord(1)-1 x_coord(end)+1 0 1], 'square')
    xticks(x_coord(1):1:x_coord(end))
    yticks(0:0.25:1)
    xlabel('# easy distractors');
    ylabel('Proportion choices [easy target]');
    box off

end