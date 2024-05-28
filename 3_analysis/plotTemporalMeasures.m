function plotTemporalMeasures(exper, anal, fixationSubset, planningTime, inspectionTime, dwellTime, responseTime, blinkFlag, nTrials)

    % Plots temporal measures for individual subjects
    %
    % The following measures are plotted:
    % - planning times (trialwise)
    % - inspection times (trialwise)
    % - dwell times (gaze shift wise)
    % - response times (trialwise)
    %
    % NOTE:
    % Plot is invisible and will not show during plotting
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % anal:
    % structure; various analysis settings, as returned by the
    % "settings_analysis" script
    %
    % fixationSubset:
    % matrix; subset of non-excluded fixations
    %
    % planningTime:
    % matrix; trialwise planning times of subjects
    %
    % inspectionTime:
    % matrix; trialwise inspection times of subjects
    %
    % dwellTime:
    % matrix; gaze-shift-wise dwell times of subjects
    %
    % responseTime:
    % matrix; trialwise response times of subjects
    %
    % blinkFlag:
    % matrix; gaze-shift-wise flags for whether a gaze shift was a blink or
    % saccade
    %
    % nTrials:
    % matrix; number of completed trials per participant and condition
    %
    % Output
    % --
    
    %% Define settings for visuals
    opt.line.style = '-';
    opt.line.color = [0, 0, 0];
    opt.line.width = 3;
    
    opt.swarm.markerSize = 120;
    opt.swarm.markerFill = 'filled';
    opt.swarm.markerFaceColor = [0, 0, 0];
    opt.swarm.markerEdgeColor = [1, 1, 1];
    opt.swarm.markerFaceAlpha = 0.15;
    
    %% Make plots
    close all;
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.fixations.subset = fixationSubset{thisSubject.number,c};
            if ismember(thisSubject.number, anal.excludedSubjects) | ...
               isempty(thisSubject.fixations.subset)
                continue
            end
            thisSubject.time.planning = planningTime{thisSubject.number,c};
            thisSubject.time.inspection = inspectionTime{thisSubject.number,c};
            thisSubject.time.dwell = dwellTime{thisSubject.number,c}(thisSubject.fixations.subset);
            thisSubject.time.response = responseTime{thisSubject.number,c};
            thisSubject.gazeShifts.isBlink = logical(blinkFlag{thisSubject.number,c}(thisSubject.fixations.subset,3));
            thisSubject.gazeShifts.n = numel(thisSubject.time.dwell);
            thisSubject.nTrials = nTrials(thisSubject.number,c);
    
            handleFigure = figure('Units', 'normalized', ...
                                  'Position', [0, 0, 1, 0.50], ...
                                  'Visible', 'off');
            tiledlayout(1,4);
    
            % Planning time
            nexttile;
            xCoord = randn(1, thisSubject.nTrials);
            swarmchart(xCoord, ..., 
                       thisSubject.time.planning, ...
                       opt.swarm.markerSize, ...
                       opt.swarm.markerFill, ...
                       'MarkerFaceColor', opt.swarm.markerFaceColor, ...
                       'MarkerEdgeColor', opt.swarm.markerEdgeColor, ...
                       'MarkerFaceAlpha', opt.swarm.markerFaceAlpha, ...
                       'HandleVisibility', 'off');
            xticks(0:10:20);
            xticks([]);
            ylabel('Trialwise planning time [ms]');
    
            % Inspection time
            nexttile;
            xCoord = randn(1, thisSubject.nTrials);
            swarmchart(xCoord, ...
                       thisSubject.time.inspection, ...
                       opt.swarm.markerSize, ...
                       opt.swarm.markerFill, ...
                       'MarkerFaceColor', opt.swarm.markerFaceColor, ...
                       'MarkerEdgeColor', opt.swarm.markerEdgeColor, ...
                       'MarkerFaceAlpha', opt.swarm.markerFaceAlpha, ...
                       'HandleVisibility', 'off');
            xticks([]);
            ylabel('Trialwise inspection time [ms]');
    
            % Dwell time
            nexttile;
            xCoord = randn(1, thisSubject.gazeShifts.n);
            swarmchart(xCoord(~thisSubject.gazeShifts.isBlink), ...
                       thisSubject.time.dwell(~thisSubject.gazeShifts.isBlink), ...
                       opt.swarm.markerSize, ...
                       opt.swarm.markerFill, ...
                       'MarkerFaceColor', opt.swarm.markerFaceColor, ...
                       'MarkerEdgeColor', opt.swarm.markerEdgeColor, ...
                       'MarkerFaceAlpha', opt.swarm.markerFaceAlpha, ...
                       'HandleVisibility', 'off');
            hold on
            swarmchart(xCoord(thisSubject.gazeShifts.isBlink), ...
                       thisSubject.time.dwell(thisSubject.gazeShifts.isBlink), ...
                       opt.swarm.markerSize, ...
                       opt.swarm.markerFill, ...
                       'MarkerFaceColor', opt.swarm.markerFaceColor, ...
                       'MarkerEdgeColor', [0, 0, 0], ...
                       'LineWidth', 2, ...
                       'MarkerFaceAlpha', opt.swarm.markerFaceAlpha);
            hold off
            xticks([]);
            ylabel('Gaze-shift-wise dwelltime [ms]');
            legend('Blink', ...
                   'Location', 'NorthWest', ...
                   'Box', 'off');
    
            % Response time
            nexttile;
            xCoord = randn(1, thisSubject.nTrials);
            swarmchart(xCoord, ...
                       thisSubject.time.response, ...
                       opt.swarm.markerSize, ...
                       opt.swarm.markerFill, ...
                       'MarkerFaceColor', opt.swarm.markerFaceColor, ...
                       'MarkerEdgeColor', opt.swarm.markerEdgeColor, ...
                       'MarkerFaceAlpha', opt.swarm.markerFaceAlpha, ...
                       'HandleVisibility', 'off');
            xticks(0:10:10);
            xticks([]);
            ylabel('Trialwise response time [ms]');
    
            set(findall(gcf, '-property', 'FontSize'), 'FontSize', 25);
            exportgraphics(handleFigure, ...
                           strcat(exper.path.figures.singleSubjects{c}, ...
                                  'temporalPerformance', ...
                                  '_s', num2str(thisSubject.number), ...
                                  '.png'), ...
                           'Resolution', 300);
            close all;
            clear thisSubject
        end
    end
end