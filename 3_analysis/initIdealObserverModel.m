function idealObserver = initIdealObserverModel(accuracy, inspectionTime, nonSearchTime)

    % Sets up option structure, passed as input to the ideal observer model
    %
    % Assumptions for ideal observer model: 
    % - each trial in the double-target condition contained ten stimuli
    % - to find a target, participants needed, on average, to fixate half 
    %   of the elements from the set of the chosen target
    % - participants received two cents for correct target discriminations, 
    %   and lost two cents for incorrect ones
    % - choice behavior was "ideal", i.e., participants always chose target
    %   with the higher gain, and behavior was NOT corrupted by noise
    %
    % Input
    % accuracy:
    % structure with fields "easy" and "difficult"; discrimination
    % performance for the two target options
    %
    % nonSearchTime:
    % structure with fields "easy" and "difficult"; nonSearchTime time for
    % the two target options
    %
    % inspectionTime:
    % structure with fields "easy" and "difficult"; inspection time for the
    % two target options
    %
    % Output
    % idealObserver:
    % structure with fields "params", "payoff", noise, and "input";
    % parameter of the experiment, required to generate model predictions, 
    % payoff matrix for correct/incorrect target discrimination, parameter
    % for additive noise, added to gain estimtes when predicting choice
    % behavior, and empirical data, required to generate model predictions

    %% Setup ideal observer model
    % Overall number of stimuli per trial (constant) as well as number of
    % distractors in each stimulus set for different set size conditions
    idealObserver.params.nStimuli = 10;
    idealObserver.params.setSizes(:,1) = 1:(idealObserver.params.nStimuli - 1);
    idealObserver.params.setSizes(:,2) = idealObserver.params.nStimuli - ...
                                         idealObserver.params.setSizes(:,1);

    % Number of fixations required to find a target
    idealObserver.params.nFixations.easy = ...
        (idealObserver.params.setSizes(:,1) - 1) / 2;
    idealObserver.params.nFixations.difficult = ...
        (idealObserver.params.setSizes(:,2) - 1) / 2;

    % Gain/loss for correct/incorrect responses
    idealObserver.payoff = [2, -2];

    %% Define noise parameter for prediction of choice behavior
    idealObserver.noise.sd = 0;
    idealObserver.noise.nSamples = 100000;

    %% Define model input, and convert temporal measures to seconds
    idealObserver.input.accuracy.easy = accuracy.easy;
    idealObserver.input.accuracy.difficult = accuracy.difficult;
    idealObserver.input.nonSearchTime.easy = nonSearchTime.easy ./ 1000;
    idealObserver.input.nonSearchTime.difficult = nonSearchTime.difficult ./ 1000;
    idealObserver.input.inspectionTime.easy = inspectionTime.easy ./ 1000;
    idealObserver.input.inspectionTime.difficult = inspectionTime.difficult ./ 1000;
end