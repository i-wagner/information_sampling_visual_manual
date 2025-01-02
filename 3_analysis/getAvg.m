function [variableAvg, variableSetSizes] = getAvg(exper, anal, variable, chosenTarget, targetId, nDistractors, subset, avgSetSize, avg)

    % Calculates the average across some variable of interest
    %
    % NOTE 1:
    % Average is calculated by, first, calculating the average of the
    % variable of interest for each set-size (i.e., numbers of distractors 
    % from the easy/difficultgiven set) seperately, and second, averaging 
    % over the resulting vector. We are doing this to be in line with the 
    % way averages are calculated in the modelling module
    %
    % NOTE 2:
    % The function assumes that invalid trials are already excluded from
    % the variable of interest (i.e., they are set to NaN), when
    % calculating the mean
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % anal:
    % structure; vairous analysis settings, as returned by the
    % "settings_analysis" script
    % 
    % variable:
    % matrix; variable of interest over which to average
    % 
    % chosenTarget:
    % matrix; ID of chosen target in trial
    %
    % targetId:
    % integer; ID of target for which to calculate discrimination
    % performance
    % 
    % nDistractors:
    % matrix; number of distractors from the set of the target, for which
    % we are calculating discrimination performance
    %
    % subset (OPTIONAL INPUT):
    % matrix; subset of entries in "variable" to use for averaging
    %
    % avgSetSize:
    % string; averaging function to use to calculate average, when 
    % calcuting avearges for differen set sie conditions. Can be "mean" or 
    % "median"
    %
    % avg:
    % string; averaging function to use to calculate average over different
    % set size conditions. Can be "mean" or "median"
    %
    % Output
    % variableAvg:
    % matrix; average of variable of interest across conditions and
    % participants

    %% Check input
    assert(ismember(avgSetSize, {'mean', 'median'}));
    assert(ismember(avg, {'mean', 'median'}));

    %% Calculate average
    variableSetSizes = [];
    variableAvg = NaN(exper.n.SUBJECTS,exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            if ismember(thisSubject.number, anal.excludedSubjects) | ...
               isempty(variable{thisSubject.number,c})
                continue
            end
            thisSubject.variable = variable{thisSubject.number,c};
            thisSubject.chosenTarget = chosenTarget{thisSubject.number,c};
            thisSubject.nDistractors = nDistractors{thisSubject.number,c};
            % Hacky way to get set-size levels:
            % when excluding even/odd trials, certain set-size levels might
            % be missing if a participant has only very few trials to being
            % with. To get around this we determine set-size level based on
            % data of all participants, instead of the current participant
            % only (which is the method used in other scripts/functions)
            thisSubject.setSize.level = unique(vertcat(nDistractors{:}));
            thisSubject.setSize.level = ...
                thisSubject.setSize.level(~isnan(thisSubject.setSize.level));
            thisSubject.setSize.n = numel(thisSubject.setSize.level);
            thisSubject.isExcludedTrial = isnan(thisSubject.variable);
            if ~isempty(subset)
                thisSubject.variable = thisSubject.variable(subset{thisSubject.number,c});
                thisSubject.nDistractors = thisSubject.nDistractors(subset{thisSubject.number,c});
                thisSubject.isExcludedTrial = thisSubject.isExcludedTrial(subset{thisSubject.number,c});
            end

            thisSubject.set = NaN(thisSubject.setSize.n, 1);
            for n = 1:thisSubject.setSize.n % Set size
                isSetSize = thisSubject.nDistractors == thisSubject.setSize.level(n);
                isTarget = true(numel(thisSubject.nDistractors), 1);
                if ~isempty(targetId)
                    isTarget = thisSubject.chosenTarget == targetId;
                end
                isTrial = isSetSize & isTarget & ~thisSubject.isExcludedTrial;

                if strcmp(avgSetSize, 'mean')
                    thisSubject.set(n) = mean(thisSubject.variable(isTrial), 1, 'omitnan');
                elseif strcmp(avgSetSize, 'median')
                    thisSubject.set(n) = median(thisSubject.variable(isTrial), 1, 'omitnan');
                end
            end
            variableSetSizes(thisSubject.number,:,c) = thisSubject.set';
            if strcmp(avg, 'mean')
                variableAvg(thisSubject.number,c) = mean(thisSubject.set, 'omitnan');
            elseif strcmp(avg, 'median')
                variableAvg(thisSubject.number,c) = median(thisSubject.set, 'omitnan');
            end
            clear thisSubject
        end
        variableSetSizes(all(variableSetSizes(:,:,c) == 0, 2),:,c) = NaN;
    end
end