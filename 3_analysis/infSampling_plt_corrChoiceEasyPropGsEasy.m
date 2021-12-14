function infSampling_plt_corrChoiceEasyPropGsEasy(propChoiceEasy_propGsEasy, plt)

    % Plots correlation between proportion choices for easy target as a
    % function of set-size and proportion choices for difficult target as a
    % function of set size
    % Input
    % propChoiceEasy_propGsEasy: matrix with proportion choices for easy 
    %                            target as a function of set-size (:, 1) and
    %                            proportion gaze shifts to easy target as a
    %                            function of set-size (:, 2), both of all
    %                            subjects
    % plt:                       structure with general plotting variables
    % Output
    % --

    %% Calculate correlation
    % Between proportion choices for easy target as a function of set-size
    % and proportion gaze shifts on easy target as a function of set-size
    [r, p] = corrcoef(propChoiceEasy_propGsEasy(:, 1), propChoiceEasy_propGsEasy(:, 2), ...
                      'Rows', 'complete');
    if p(2) > 0.05

        sig_star = 'ns';

    elseif p(2) <= 0.001

        sig_star = '***';

    elseif p(2) <= 0.01

        sig_star = '**';

    elseif p(2) <= 0.05

        sig_star = '*';

    end


    %% Plot
    xBound = [-0.05 1.05];
    yBound = [-0.05 1.05];

    plot(propChoiceEasy_propGsEasy(:, 1), propChoiceEasy_propGsEasy(:, 2), ...
         'o', ...
         'MarkerSize',      plt.size.mrk_ss, ...
         'MarkerFaceColor', plt.color.c1, ...
         'MarkerEdgeColor', plt.color.white)
    hold on
    ls_h = lsline;
    set(ls_h, ...
        'Color',     plt.color.black, ...
        'LineWidth', plt.lw.thick)
    l_h = line([xBound(1) xBound(2)], [0.5 0.5], ...
               'LineWidth', plt.lw.thick, ...
               'LineStyle', '--', ...
               'Color',     plt.color.c1);
    uistack([ls_h l_h], 'top')
    xlabel('Proportion choices easy target')
    ylabel('Proportion gaze shifts easy set')
    axis([xBound(1) xBound(2) yBound(1) yBound(2)], 'square')
    xticks(0:0.25:1)
    yticks(0:0.25:1)
    text(0, 1, ['r = ', num2str(round(r(2), 2)), sig_star])
    box off

end