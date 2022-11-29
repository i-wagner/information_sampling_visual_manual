function infSampling_model_plots(inp_emp_propChoicesEasy, inp_predStochastic_propChoicesEasy, inp_predNoise_propChoicesEasy, ...
                                 inp_emp_perf,            inp_predStochastic_perf,            inp_predPerfect_perf, ...
                                 inp_emp_propFixChosen, inp_emp_propFixEasy, inp_predStochastic_propFixChosen, inp_predStochastic_propFixEasy, ...
                                 inp_switch_simpleComplex, plt)

    % Generates some plots with results of model fit
    % Input
    % inp_emp_propChoicesEasy:            empirical proportion choices for easy targets
    % inp_predStochastic_propChoicesEasy: model predictions (stochastic model) for proportion choices for easy targets
    % inp_predNoise_propChoicesEasy:      model predictions (noise model) for proportion choices for easy targets
    % inp_emp_perf:                       empirical average gain
    % inp_predStochastic_perf:            model predictions (stochastic model) for gain
    % inp_predPerfect_perf:               model predictions (perfect model) for gain
    % inp_emp_propFixChosen:              empirical proportion fixations on chosen set
    % inp_emp_propFixEasy:                empirical proportion fixations on easy set
    % inp_predStochastic_propFixChosen:   model predictions (stochastic model) for proportion fixations on chosen set
    % inp_predStochastic_propFixEasy:     model predictions (stochastic model) for proportion fixations on easy set
    % inp_switch_simpleComplex:           Determines if results of easy/complex model are plotted
    % plt:                                structure with some general plot settings
    % Output
    % --

    %% Init
    close all

    no_sub = size(inp_emp_propChoicesEasy, 1);
    no_ss  = size(inp_emp_propChoicesEasy, 2);
    if inp_switch_simpleComplex == 1

        pltName_suffix = '_simple';

    elseif inp_switch_simpleComplex == 2

        pltName_suffix = '_complex';

    elseif inp_switch_simpleComplex == 3

        pltName_suffix = '_complexVariableNoiseFixedBias';

    end


    %% Proprotion choices easy target
    % Empirical, predicted by model with unsystematic fixation, predicted
    % by model with decision noise
    fig_h = figure;
    for s = 3:no_sub % Subject

        nexttile(s);
        plot(0:8, inp_emp_propChoicesEasy(s, :),            '-r', ... Choices; empirical
             0:8, inp_predStochastic_propChoicesEasy(s, :), '-b', ... Choices; predicted, stochastic model
             0:8, inp_predNoise_propChoicesEasy(s, :),      '-g', ... Choices; predicted, noise model
             'LineWidth', 2);
        axis([-1 9 0 max([1, inp_predStochastic_propChoicesEasy(s, :)])])
        if s == 3

            xlabel('# easy distractors');
            ylabel('Choices easy target [proportions]');

        end
        box off
        title(['Subject ', num2str(s)]);

    end
    leg_h = legend({'Empirical'; 'Predicted (stochastic)'; 'Predicted (noise)'});
    leg_h.Layout.Tile = 1;
    legend boxoff
    opt.size    = [35 35];
    opt.imgname = strcat('propChoicesEasy_subject', pltName_suffix);
    opt.save    = 1;
    prepareFigure(fig_h, opt)
    close


    %% Predicted vs empirical performance/porportion choices easy target
    infSampling_plt_figPerf(inp_emp_perf, inp_predPerfect_perf, inp_predStochastic_perf, ...
                            inp_emp_propChoicesEasy, inp_predStochastic_propChoicesEasy, plt)


    %% Overall proportions fixations on chosen/easy set
    propFix_pred_mean = [squeeze(mean(inp_predStochastic_propFixChosen, 2)), ...
                         squeeze(mean(inp_predStochastic_propFixEasy, 2))];
    propFix_emp_mean  = [squeeze(mean(inp_emp_propFixChosen, 2)), ...
                         squeeze(mean(inp_emp_propFixEasy, 2))];
    no_sp             = size(propFix_pred_mean, 2);
    labX_sp           = {'Proportion fixations chosen set [empirical]'; ...
                         'Proportion fixations easy set [empirical]'};
    labY_sp           = {'Proportion fixations chosen set [predicted]'; ...
                         'Proportion fixations easy set [predicted]'};

    fig_h = figure;
    for sp = 1:no_sp % Subplot

        [rSquared, p] = corrcoef(propFix_pred_mean(:, sp), propFix_emp_mean(:, sp), 'Rows', 'Complete');
        rSquared      = round(rSquared(1, 2).^2, 2);

        nexttile
        scatter(propFix_emp_mean(:, sp), propFix_pred_mean(:, sp), ...
                'MarkerFaceColor', [0 0 0], ...
                'MarkerEdgeColor', [1 1 1])
        line([0 1], [0 1])
        text(0.10, 0.90, ['R^2 = ' num2str(rSquared), ' p = ', num2str(round(p(1, 2), 3))]);
        axis square
        xticks(0:0.25:1)
        yticks(0:0.25:1)
        xlabel(labX_sp{sp})
        ylabel(labY_sp{sp})

    end
    opt.size    = [35 15];
    opt.imgname = strcat('propFixationsOnChosen_all', pltName_suffix);
    opt.save    = 1;
    prepareFigure(fig_h, opt)
    close


    %% Proportion fixations on chosen/easy set as a function of set size
    % Seperate for each subject
    propFix_pred = cat(3, ...
                       inp_predStochastic_propFixChosen, ...
                       inp_predStochastic_propFixEasy);
    propFix_emp  = cat(3, ...
                       inp_emp_propFixChosen, ...
                       inp_emp_propFixEasy);
    labY_sp      = {'Fixations on chosen [proportion]'; ...
                    'Fixations on easy [proportion]'};
    name_sp      = {'Chosen'; 'Easy'};
    no_sp        = 1;
    for sp = 1:no_sp % Subplot

        fig_h = figure;
        for s = 3:no_sub % Subject

            nexttile(s);
            hold on
            line([4 4], [0 1], 'HandleVisibility', 'Off')
            plot(0:8, propFix_emp(s, :, sp),  '-r', ... Empirical
                 0:8, propFix_pred(s, :, sp), '-b', ... Predicted
                 'LineWidth', 2);
            hold off
            axis([-1 9 0 1])
            xticks(0:1:8)
            if s == 3

                xlabel('# easy distractors')
                ylabel(labY_sp{sp})

            end
            title(['Subject ', num2str(s)]);

        end
        leg_h = legend({'Empirical'; 'Predicted'});
        leg_h.Layout.Tile = 1;
        legend boxoff
        opt.size    = [35 35];
        opt.imgname = strcat('propFixationsOn', name_sp{sp}, '_subjects', pltName_suffix);
        opt.save    = 1;
        prepareFigure(fig_h, opt)
        close

    end


    %% Proportion fixations on chosen/easy set as a function of set size
    % Seperate for each set size
    propFix_pred = cat(3, ...
                       inp_predStochastic_propFixChosen, ...
                       inp_predStochastic_propFixEasy);
    propFix_emp  = cat(3, ...
                       inp_emp_propFixChosen, ...
                       inp_emp_propFixEasy);
    labX_sp      = {'Proportion fixations chosen set [empirical]'; ...
                    'Proportion fixations easy set [empirical]'};
    labY_sp      = {'Proportion fixations chosen set [predicted]'; ...
                    'Proportion fixations easy set [predicted]'};
    name_sp      = {'Chosen'; 'Easy'};
    no_sp        = 1;
    for sp = 1:no_sp % Subplot

        fig_h = figure;
        for ss = 1:no_ss % Set size

            nexttile(ss);
            scatter(propFix_emp(:, ss, sp), propFix_pred(:, ss, sp), ...
                    'MarkerFaceColor', [0 0 0], ...
                    'MarkerEdgeColor', [1 1 1])
            line([0 1], [0 1])
            axis square
            if ss == 1

                xlabel(labX_sp{sp})
                ylabel(labY_sp{sp})

            end
            title([num2str(ss)-1, ' easy distractors']);

        end
        opt.size    = [45 35];
        opt.imgname = strcat('propFixationsOn', name_sp{sp}, '_setSize', pltName_suffix);
        opt.save    = 1;
        prepareFigure(fig_h, opt)
        close

    end

end