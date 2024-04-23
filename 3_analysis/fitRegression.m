function linRegFit = fitRegression(exper, anal, proportionEasyChoices, nEasyDistractors)

    % Fit a linear regression to choice data of participants, and extract
    % slope and intercepts
    %
    % NOTE:
    % Regressions are fitted to normalised data, i.e., proportion choices
    % and distractors in easy set were centered on the respective axis
    % before fitting. By doing this, slopes and intercepts provide an
    % estimate for how much whether participants preferred one over the
    % other target (intercept), and when participants changed their target
    % preference
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
    % proportionEasyChoices:
    % matrix; proportion choices for easy target for different set sizes of
    % easy distractors
    % 
    % nEasyDistractors:
    % matrix; trialwise number of easy
    %
    % Output
    % linRegFit:
    % matrix; intercepts (:,1,:) and slopes (:,2,:) for participants across
    % conditions

    %% Fit regression
    linRegFit = NaN(exper.n.SUBJECTS, 2, exper.n.CONDITIONS);
    for c = 1:exper.n.CONDITIONS % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            if ismember(thisSubject.number, anal.excludedSubjects)
                continue
            end
            thisSubject.nEasyDistractors = nEasyDistractors{thisSubject.number,c};
            thisSubject.setSize.level = ...
                unique(thisSubject.nEasyDistractors(~isnan(thisSubject.nEasyDistractors)));
            thisSubject.proportionEasyChoices = proportionEasyChoices(thisSubject.number,:,c)';

            % x: subtract half thre number of easy distractors to center
            %    curve along x-axis
            % y: subtract chance performance to center curve along y-axis
            thisSubject.nHalfDistractors = max(thisSubject.setSize.level) / 2;
            thisSubject.chancePerformance = 0.50;
            thisSubject.x = thisSubject.setSize.level - thisSubject.nHalfDistractors;
            thisSubject.y = thisSubject.proportionEasyChoices - thisSubject.chancePerformance;
            if ~isempty(thisSubject.x)
                linRegFit(thisSubject.number,:,c) = ...
                    regress(thisSubject.y, [ones(size(thisSubject.x)), thisSubject.x]);
            end
            clear thisSubject
        end
    end
end