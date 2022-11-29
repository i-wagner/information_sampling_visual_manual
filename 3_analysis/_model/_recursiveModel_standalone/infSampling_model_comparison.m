function infSampling_model_comparison(inp_par_simple, inp_par_complex, inp_weights_simple, inp_weights_complex, inp_noDp, plt)

    % Plots parameter distribution and model weights
    %
    % Input
    % inp_par_simple/_complex:     parameter distributions for simple/complex
    %                              model; rows are subjects; simple model has
    %                              one, complex model 2 free parameter
    % inp_weights_simple/_complex: sum of squared residuals for simple/complex
    %                              model
    % plt:                        structure with general plot settings
    %
    % Output
    % --

    %% Parameter distribution
    fig_h = figure;
    tiledlayout(1, 2)
    nexttile;
    plot(1:2, [inp_par_simple, inp_par_complex(:, 1)], ...
         '-o', ...
         'MarkerFaceColor', [0 0 0], ...
         'MarkerEdgeColor', [1 1 1], ...
         'Color',           [0 0 0])
    axis([0 3 0 1])
    xticks(1:1:2)
    xticklabels({'Simple'; 'Complex'});
    xlabel('Model')
    ylabel('Parameter value')
    title('SD of Gaussian')
    box off

    nexttile;
    plot(1:2, [NaN(numel(inp_par_simple), 1), inp_par_complex(:, 2)], ...
         '-o', ...
         'MarkerFaceColor', [0 0 0], ...
         'MarkerEdgeColor', [1 1 1], ...
         'Color',           [0 0 0])
    axis([0 3 0 2])
    xticks(1:1:2)
    xticklabels({'Simple'; 'Complex'});
    xlabel('Model')
    title('Decision noise [10000 samples]')
    box off
    opt.size    = [30 15];
    opt.imgname = strcat('parDist');
    opt.save    = 1;
    prepareFigure(fig_h, opt)
    close; clear fig_h


    %% Model weights
    fig_h = figure;
    line([[0; 1], [0; 1], [0.50; 0.50]], [[0; 1], [0.50; 0.50], [0; 1]], ...
         'Color', [0 0 0])
    hold on
    scatter(inp_weights_simple, inp_weights_complex, ...
            'MarkerFaceColor', [0 0 0], ...
            'MarkerEdgeColor', [1 1 1])
    hold off
    axis([0 1 0 1], 'square')
    xticks(0:0.25:1)
    yticks(0:0.25:1)
    xlabel('Model weights [simple model]')
    ylabel('Model weights [complex model]')
    text(inp_weights_simple(inp_weights_complex > inp_weights_simple) + 0.02, ...
         inp_weights_complex(inp_weights_complex > inp_weights_simple) - 0.02, ...
         num2str(find(inp_weights_complex > inp_weights_simple)));
    opt.size    = [15 15];
    opt.imgname = strcat('infoWeights');
    opt.save    = 1;
    prepareFigure(fig_h, opt)
    close
    clear fig_h

end