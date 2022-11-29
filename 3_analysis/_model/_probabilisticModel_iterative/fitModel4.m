close all


%% Init
emp_propChoicesEasy       = stim.propChoice.easy(:, :, 2);                           % Empirical proportion choices easy target
emp_propFixChosenSet      = mean(sacc.propGs.onAOI_modelComparision2(:, :, 2), 2, 'omitnan'); %sacc.propGs.onAOI_modelComparision(:, :, 2);             % Mean proportion fixations on chosen/non-chosen set
emp_propFixChosenSet_ss   = sacc.propGs.onAOI_modelComparision2(:, :, 2);            % Set-size-wise proportion fixations on chosen set
emp_propFixEasySet_ss     = sacc.propGs.onAOI_modelComparision_easyDiff_ss(:, :, 2); % Set-size-wise proportion fixations on easy set
emp_perf                  = all.data.double.perf;                                    % Gain per unit of time (empirical)

noSs  = size(emp_propChoicesEasy, 1); % # set sizes
noSub = size(emp_propChoicesEasy, 2); % # subjects


%% Fitting options
model.minSearch.options = optimset('MaxFunEvals', 10000, ...
                                   'MaxIter',     10000, ...
                                   'TolFun',      1e-12, ...
                                   'TolX',        1e-12);
model.minSearch.init    = 0.50;
model.minSearch.min     = 0;
model.minSearch.max     = 1;


%% Empirical gain
% emp_gain = all.model.gain(:, :, :, 3);
emp_gain = NaN(noSub, noSs, 2);
for s = 1:noSub % Subject

    for t = 1:2 % Easy/difficult target

        emp_gain(s, :, t) = ...
            (perf.hitrates(s, 1, t+1) .* all.params.payoff(1, t) + (1 - perf.hitrates(s, 1, t+1)) .* all.params.payoff(2, t)) ./ ...
            (all.data.pred.mean_item_per_set(s, :, t) .* (sacc.time.mean.inspection(s, 1, t+1)/1000) + (sacc.time.mean.non_search(s, 1, t+1)/1000));

    end
    clear t

end


%% Fit free parameter
freeParameter = NaN(noSub, 1);
parfor s = 3:noSub % Subject

    setSizes = unique([stim.no_easyDis{s, 2}+1 stim.no_hardDis{s, 2}+1], 'rows');
    setSizes = setSizes(any(~isnan(setSizes), 2), :);

    freeParameter(s) = fminsearchbnd(@lossFunction, ...
                                     model.minSearch.init, ...
                                     model.minSearch.min, ...
                                     model.minSearch.max, ...
                                     model.minSearch.options, ...
                                     squeeze(emp_gain(s, :, :)), ...
                                     emp_propChoicesEasy(:, s), ...
                                     emp_propFixChosenSet_ss(s, :), ...
                                     emp_propFixEasySet_ss(s, :), ...
                                     setSizes);

end


%% Predict proportion choices easy target + fixations on sets
close all

pred_propChoices = NaN(noSs, 2, noSub);
pred_noFix       = NaN(noSs, 3, 2, noSub);
pred_gain        = NaN(noSs, 2, noSub);
pred_perf        = NaN(noSub, 1);

fig_h = figure;
for s = 3:noSub % Subject

    setSizes = unique([stim.no_easyDis{s, 2}+1 stim.no_hardDis{s, 2}+1], 'rows');
    setSizes = setSizes(any(~isnan(setSizes), 2), :);

    % Predict proportion choices easy target
    [pred_propChoices(:, :, s), pred_noFix(:, :, :, s)] = ...
        test_unsystematic_fixations_2(setSizes, squeeze(emp_gain(s, :, :)), freeParameter(s));

    % Predict
    for t = 1:2 % Easy/difficult target

        pred_gain(:, t, s) = ...
            (perf.hitrates(s, 1, t+1) .* all.params.payoff(1, t) + (1 - perf.hitrates(s, 1, t+1)) .* all.params.payoff(2, t)) ./ ...
            (pred_noFix(:, t, 1, s) .* (sacc.time.mean.inspection(s, 1, t+1)/1000) + (sacc.time.mean.non_search(s, 1, t+1)/1000));

    end
    pred_perf(s) = mean((pred_propChoices(:, 1, s) .* pred_gain(:, 1, s)) + ...
                        (pred_propChoices(:, 2, s) .* pred_gain(:, 2, s)));

    % Plot proprotion choices easy target
    nexttile(s);
    plot(0:8, emp_propChoicesEasy(:, s),  '-r', ... % Choices; empirical
         0:8, pred_propChoices(:, 1, s), '-b', ...  % Choices; predicted, fixations model
         0:8, all.model.choices(s, :, 3), '-g', ... % Choices; predicted, noise model
         'LineWidth', 2);
    axis([-1 9 0 max([1; pred_propChoices(:, 1, s)])+0.10])
    if s == 3

        xlabel('# easy distractors');
        ylabel('Choices easy target [proportions]');
    end
    box off
    title(['Subject ', num2str(s)]);

end
leg_h = legend({'Empirical'; 'PredictedNew'; 'PredictedNoise'});
leg_h.Layout.Tile = 1;
legend boxoff
opt.size    = [35 35];
opt.imgname = strcat('propChoicesEasy_subject');
opt.save    = 1;
prepareFigure(fig_h, opt)
close


%% Predicted vs empirical performance/porportion choices easy target
pred_perf_perfect = all.model.perf_perfect(:, 3);

fig_h = figure;
infSampling_plt_figFour(emp_perf, pred_perf_perfect, pred_perf, emp_propChoicesEasy', squeeze(pred_propChoices(:, 1, :))', plt)
sublabel([], -10, -25);
opt.size    = [45 15];
opt.imgname = strcat('performance');
opt.save    = 1;
prepareFigure(fig_h, opt)
close


%% Predicted vs empirical # fixated distractors
pred_propFixSet = squeeze(mean(pred_noFix(:, 1:2, 2, :) ./ pred_noFix(:, 3, 2, :), 1))'; % On chosen/not-chosen
% pred_propFixSet = squeeze(mean(pred_noFix(:, 1:2, 1, :) ./ pred_noFix(:, 3, 1, :), 1))'; % On easy/difficult

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot proportion fixations chosen as a function of set size
pred_propFixSet2 = squeeze(pred_noFix(:, 1:2, 2, :) ./ pred_noFix(:, 3, 2, :)); % On chosen/not-chosen
pred_propFixSet3 = squeeze(pred_noFix(:, 1:2, 1, :) ./ pred_noFix(:, 3, 1, :)); % On easy/difficult

fig_h = figure;
for s = 3:noSub % Subject

    nexttile(s);
    hold on
    line([4 4], [0 1], 'HandleVisibility', 'Off')
    plot(0:8, emp_propFixChosenSet_ss(s, :)',  '-r', ... % Empirical
         0:8, pred_propFixSet2(:, 1, s), '-b', ...       % Predicted
         'LineWidth', 2);
    hold off
    axis([-1 9 0 1])
    xticks(0:1:8)
    if s == 3
        xlabel('# easy distractors')
        ylabel('Fixations on chosen [proportion]')
    end
    title(['Subject ', num2str(s)]);

end
leg_h = legend({'Empirical'; 'Predicted'});
leg_h.Layout.Tile = 1;
legend boxoff

opt.size    = [35 35];
opt.imgname = strcat('propFixationsOnChosen_subjects');
opt.save    = 1;
prepareFigure(fig_h, opt)
close
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [bls, ~, ~, ~, stats_linreg] = regress(pred_propFixSet(:, 1), [ones(21, 1) emp_propFixSet(:, 1)]);
% [brob, stats]                = robustfit(emp_propFixSet(:, 1), pred_propFixSet(:, 1));
% 
% outliers_ind = find(abs(stats.resid) > stats.mad_s);
% 
% rquare_linreg     = stats_linreg(1);
% rsquare_robustfit = corr(pred_propFixSet(:, 1), brob(1)+brob(2)*emp_propFixSet(:, 1), 'Rows', 'Complete')^2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[rSquared, p] = corrcoef(pred_propFixSet(:, 1), emp_propFixChosenSet(:, 1), 'Rows', 'Complete');
rSquared      = round(rSquared(1, 2).^2, 2);

fig_h = figure;
scatter(emp_propFixChosenSet(:, 1), pred_propFixSet(:, 1), ...
        'MarkerFaceColor', [0 0 0], ...
        'MarkerEdgeColor', [1 1 1])
line([0 1], [0 1])
text(0.10, 0.90, ['R^2 = ' num2str(rSquared), ' p = ', num2str(round(p(1, 2), 3))]);
axis square
xlabel('Proportion fixations chosen set [empirical]')
ylabel('Proportion fixations chosen set [predicted]')
opt.size    = [15 15];
opt.imgname = strcat('propFixationsOnChosen_all');
opt.save    = 1;
prepareFigure(fig_h, opt)
close

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot proportion fixations chosen as a function of set size
% Seperate for set sizes
fig_h = figure;
for ss = 1:size(emp_propFixChosenSet_ss, 2) % Set size

    nexttile(ss);
    scatter(emp_propFixChosenSet_ss(:, ss), squeeze(pred_propFixSet2(ss, 1, :)), ...
            'MarkerFaceColor', [0 0 0], ...
            'MarkerEdgeColor', [1 1 1])
    line([0 1], [0 1])
    axis square
    if ss == 1
        xlabel('Proportion fixations chosen set [empirical]')
        ylabel('Proportion fixations chosen set [predicted]')
    end
    title([num2str(ss)-1, ' easy distractors']);

end
opt.size    = [45 35];
opt.imgname = strcat('propFixationsOnChosen_setSize');
opt.save    = 1;
prepareFigure(fig_h, opt)
close
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%