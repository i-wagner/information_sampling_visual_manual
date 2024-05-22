function gainPerTime = getGain(accuracy, inspectionTime, nonSearchTime, avgItemsPerSet, payoff)

    % Calculates monetary gain per unit of time for target option
    %
    % Input
    % accuracy:
    % double; discrimination accuracy for target
    %
    % inspectionTime:
    % double, HAS TO BE IN SECONDS; average inspection time when target 
    % option was chosen
    %
    % nonSearchTime:
    % double, HAS TO BE IN SECONDS; average non-search time (i,e.,
    % planning + response time) when target option was chosen
    %
    % avgItemsPerSet:
    % vector; average number of stimuli from the same set as the target
    % that need to fixated, before the target is found
    %
    % payoff:
    % vector; payoff for correct (1) and incorrect (2) target
    % discriminations
    % 
    % Output
    % gainPerTime:
    % vector; gain estimate for target option, given empirical data. Gain
    % per time is returned for each set size seperately

    %% Calculate monetary gain per unit of time
    gain = accuracy * payoff(1) + (1 - accuracy) * payoff(2);
    time = (avgItemsPerSet .* inspectionTime) + nonSearchTime;
    gainPerTime = gain ./ time;
end