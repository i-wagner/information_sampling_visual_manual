% Replicates the proportion gaze shifts on AOI analysis from the old
% analysis pipeline (the one with the ugly bar graphs)


%% 
no_subs = size(sacc.gazeShifts, 1);

propGsInCat_singleSub           = cell(no_subs, 1);
propGsInCat_target_ss_singleSub = cell(no_subs, 1);
for s = 1:no_subs

    dat = sacc.gazeShifts{s, 2};
    if ~isempty(dat)

        % Get data
        setSizes = stim.no_easyDis{s, 2};               % # easy distractors
        no_ss    = unique(setSizes(~isnan(setSizes)));  %
        targChoi = stim.chosenTarget{s, 2};             % Chosen target
        stim_cat = unique(dat(~isnan(dat(:, 18)), 18)); % Stimulus categories
        aois     = dat(:, 18);                          % Target AOIs
        trialNos = dat(:, 26);                          % Trial numbers
        clear dat

        % Trialwise proportion gaze shifts to AOI
        propGsInCat = NaN(max(trialNos), numel(stim_cat));
        for t = 1:max(trialNos) % Trial

            li_trial = trialNos == t;
            no_gs    = sum(li_trial);
            for sc = 1:numel(stim_cat) % Stimulus category

                li_cat = aois(li_trial) == stim_cat(sc);

                propGsInCat(t, sc) = sum(li_cat) / no_gs;
                clear li_cat

            end
            clear li_trial no_gs

        end
        propGsInCat_singleSub{s} = propGsInCat;
        clear aois aois

        % Mean proportion gaze shifts to AOI, seperate for different set-size
        % level and target choices
        propGsInCat_target_ss = NaN(numel(no_ss), numel(stim_cat), 2);
        for d = 1:2 % Chosen target
    
            for ss = 1:numel(no_ss) % # easy distractors
    
                trials = propGsInCat(targChoi == d & setSizes == no_ss(ss), :);
%                 trials = propGsInCat(targChoi == d, :);
    
                propGsInCat_target_ss(ss, :, d) = mean(trials, 'omitnan');
    
            end
    
        end
        propGsInCat_target_ss_singleSub{s} = propGsInCat_target_ss;
        clear stim_cat propGsInCat

    end

end


%%
% clc; cellfun(@(x) mean(x, 'omitnan'), propGsInCat_singleSub, 'UniformOutput', false)
% clc; cell2mat(cellfun(@(x) x(1, :, :), propGsInCat_target_ss_singleSub(3:end), 'UniformOutput', false))


%%
color_target  = {'#384B60' '#4A544A'};
color_distrct = {'#5C93C4' '#849D6A'};
color_misc    = {'#9C2E8F'};
                    
axisLimits    = [-2 11 0 1.1];
subplotTitles = {'Easy target chosen'; 'Hard target chosen'};
legendLabels  = {'Easy target'; 'Hard target'; 'Easy distractor'; 'Hard distractor'; 'Background'};
subplotFc     = [color_target color_distrct color_misc];
legendxPos    = 0.85;
legendyPos    = 0.71;

% Plot
fig_h = figure('visible', 'on');
for d = 1:2 % Target difficulty

    subplot(1, 2, d)
    b = bar(0:1:8, propGsInCat_target_ss_singleSub{3}(:, :, d), 'stacked', ...
            'EdgeColor', 'None');                                            % Proportion saccades on AOI
    for bc = 1:5 % Bar color

        b(bc).FaceColor = subplotFc{bc};

    end
    hold on
%     plot(0:8, stat.propChosenTsetSize(:, d, vpNo), '-o',...
%         'LineWidth', plt.lw, ...
%         'MarkerSize', 10, ...
%         'Color', plt.color.black, ...
%         'MarkerEdgeColor', plt.color.white, ...
%         'MarkerFaceColor', plt.color.black);                                         % Proportion trials easy target chosen
    hold off
    axis(axisLimits, 'square')
    xticks(0:2:8)
    yticks(0:0.25:1)
    xlabel('Number of easy distractors')
    if d == 1

        ylabel('Mean proportion saccades on AOI')

    end
    title(subplotTitles{d})
    box off

end
leg = legend(legendLabels);
title(leg, 'AOI');
leg.Position(1) = legendxPos;
leg.Position(2) = legendyPos;
legend boxoff