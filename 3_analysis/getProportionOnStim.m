function proportionOnStim = getProportionOnStim(nCompletedTrial, excludedTrials, trialMap, stimIds, stimId, exper)

    % Calculates proportion of fixations that targeted some stimulus of 
    % interest
    %
    % Input
    %
    %
    % Output
    % proportionOnStim:
    % 
    
    %% Get proportion fixations on background
    proportionOnStim = NaN(exper.n.SUBJECTS, exper.n.CONDITIONS, 2);
    for c = 1:exper.n.CONDITIONS
        for s = 1:exper.n.SUBJECTS
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.nTrials = ...
                nCompletedTrial(thisSubject.number,c);
            thisSubject.excludedTrials = ...
                excludedTrials{thisSubject.number,c};
            thisSubject.trialMap = trialMap{thisSubject.number,c};
            thisSubject.fixatedAois = stimIds{thisSubject.number,c};
            thisSubject.nFixations = numel(thisSubject.fixatedAois);
            thisSubject.isOnStim = thisSubject.fixatedAois == stimId;
            if isnan(thisSubject.nTrials)
                continue
            end
    
            % Averaged over all trials
            proportionOnStim(thisSubject.number,c,1) = ...
                sum(thisSubject.isOnStim) / thisSubject.nFixations;
    
            % Seperately for trials
            % Note: 
            % -- "propInTrials" is NaN if a trial was excluded
            % -- "propInTrials" is zero if the stimulus of interest was
            %    actually never fixated in a trial
            propInTrials = NaN(thisSubject.nTrials, 1);
            for t = 1:thisSubject.nTrials
                if ismember(t, thisSubject.excludedTrials)
                    continue
                end
                thisTrial.idx = thisSubject.trialMap == t;
                thisTrial.isOnStim = ...
                    thisSubject.fixatedAois(thisTrial.idx) == stimId;
    
                propInTrials(t) = ...
                    sum(thisTrial.isOnStim) / sum(thisTrial.idx);
                clear thisTrial
            end
            proportionOnStim(thisSubject.number,c,2) = ...
                mean(propInTrials, 1, 'omitnan');
            clear thisSubject
        end
    end
    
    %% Plot
    DEBUG = false;
    if DEBUG
        plotTitles = ["Over all fixation", "Over individual trials"];
        
        close all;
        figure;
        tiledlayout(1,2);
        for t = 1:2
            nexttile;
            line([0, 3], [0.50, 0.50], ...
                 'LineStyle', '-', ...
                 'Color', [0, 0, 0], ...
                 'HandleVisibility', 'off');
            hold on
            errorbar(0.75:1.50:2.25, ...
                     mean(proportionOnStim(:,1:2,t), 1, 'omitnan'), ...
                     std(proportionOnStim(:,1:2,t), 1, 'omitnan'), ...
                     'o', ...
                     'MarkerFaceColor', [0, 0, 0], ...
                     'MarkerEdgeColor', [1, 1, 1], ...
                     'Color', [0, 0, 0]);
            plot(1:2, proportionOnStim(:,1:2,t), ...
                 '-o', ...
                 'MarkerFaceColor', [0, 0, 0], ...
                 'MarkerEdgeColor', [1, 1, 1], ...
                 'Color', [0, 0, 0]);
            hold off
            axis([0, 3, 0, 1], 'square');
            xticks(1:2);
            yticks(0:0.25:1);
            xticklabels(["Single", "Double"]);
            xlabel('Visual-search condition');
            ylabel('Proportion fixations on background');
            title(plotTitles(t));
            legend(["Mean +/- SD", "Subjects"]);
            legend box off;
            box off
        end
        set(gcf, "Units", "Normalized", "Position", [0, 0, 0.50, 0.35]);
    end

end