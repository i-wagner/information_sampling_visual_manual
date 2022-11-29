function perf = infSampling_calculateGain(acc, choices, searchTime, nonSearchTime, noFix_overall, correctFix)

    % Calculate monetary gain per unit of time
    % FOR ALL INPUTS: rows are always subjects, columns are either set-sizes or
    %                 target difficulty, depending on the input
    % Input
    % acc:           overall accuracy for easy (:, 1) and difficult (:, 2) target
    % choices:       set-size-wise proportion choices for easy target
    % searchTime:    overall search time for easy (:, 1) and difficult (:, 2) target
    % nonSearchTime: overall non-search time for easy (:, 1) and difficult (:, 2) target
    % noFix_overall: set-size-wise # saccades required to find any target
    % correctFix:    switch; correct for target fixation (1) or not (0)
    % Output
    % perf:          average gain over all set sizes

    %% Init
    % Data structure
    no_sub = size(choices, 1);
    no_ss  = size(choices, 2);

    % Reward struture
    win  = 2;
    loss = -2;


    %% Correct accuracy
    % ?
%     acc(acc < 0.50) = 0.50;


    %% Calculate gain
    gain  = NaN(no_sub, no_ss);
%     gain2 = NaN(no_sub, no_ss);
%     gain3 = NaN(no_sub, no_ss);
    for s = 1:no_sub % Subject

        acc_easy           = acc(s, 1);
        acc_diff           = acc(s, 2);
        searchTime_easy    = searchTime(s, 1);
        searchTime_diff    = searchTime(s, 2);
        nonSearchTime_easy = nonSearchTime(s, 1);
        nonSearchTime_diff = nonSearchTime(s, 2);
        for ss = 1:no_ss % Set size

            choices_easy  = choices(s, ss);
            choices_diff  = (1 - choices_easy);
            noFixToTarget = noFix_overall(s, ss);
            if correctFix == 1

                gain(s, ss) = ...
                    ((acc_easy * win * choices_easy) + ((1 - acc_easy) * loss * choices_easy) + (acc_diff * win * choices_diff) + ((1 - acc_diff) * loss * choices_diff)) / ...
                    (((noFixToTarget - 1) * ((choices_easy * searchTime_easy) + (choices_diff * searchTime_diff))) + (choices_easy * nonSearchTime_easy) + (choices_diff * nonSearchTime_diff));

            elseif correctFix == 0

                gain(s, ss) = ...
                    ((acc_easy * win * choices_easy) + ((1 - acc_easy) * loss * choices_easy) + (acc_diff * win * choices_diff) + ((1 - acc_diff) * loss * choices_diff)) / ...
                    ((noFixToTarget * ((choices_easy * searchTime_easy) + (choices_diff * searchTime_diff))) + (choices_easy * nonSearchTime_easy) + (choices_diff * nonSearchTime_diff));

            end

            % DEBUG
            % Simplified formula to calculate gain
%             zaehler_easy = (acc_easy * win * choices_easy) + ((1 - acc_easy) * loss * choices_easy);
%             zaehler_diff = (acc_diff * win * choices_diff) + ((1 - acc_diff) * loss * choices_diff);
%             nenner_easy  = ((noFixToTarget - 1) * (choices_easy * searchTime_easy)) + (choices_easy * nonSearchTime_easy);
%             nenner_diff  = ((noFixToTarget - 1) * (choices_diff * searchTime_diff)) + (choices_diff * nonSearchTime_diff);

%             gain2(s, ss) = (zaehler_easy + zaehler_diff) / (nenner_easy + nenner_diff);

        end

        % DEBUG
        % Even more simplified formula to calculate gain; here, directly
        % over all set sizes
%         zaehler_easy = (acc_easy * win * choices(s, :)') + ((1 - acc_easy) * loss * choices(s, :)');
%         zaehler_diff = (acc_diff * win * (1 - choices(s, :)')) + ((1 - acc_diff) * loss * (1 - choices(s, :)'));
%         nenner_easy  = ((noFix_overall(s, :)' - 1) .* (choices(s, :)' * searchTime_easy)) + (choices(s, :)' * nonSearchTime_easy);
%         nenner_diff  = ((noFix_overall(s, :)' - 1) .* ((1 - choices(s, :)') * searchTime_diff)) + ((1 - choices(s, :)') * nonSearchTime_diff);

%         gain3(s, :) = (zaehler_easy + zaehler_diff) ./ (nenner_easy + nenner_diff);

    end
    perf = mean(gain, 2, 'omitnan');
%     all(all((round(gain(3:end, :), 10) == round(gain2(3:end, :), 10)) & (round(gain2(3:end, :), 10) == round(gain3(3:end, :), 10))))

end