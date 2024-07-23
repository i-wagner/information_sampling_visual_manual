function dev = lossCdf(par, empX, empY)

    % Calculates loss when fitting Gaussian CDF to choice data
    %
    % NOTE 1:
    % CDF is inverted by taking 1 - fit
    %
    % NOTE 2:
    % We are using sum of squared residuals as loss
    %
    % Input:
    % par:
    % vector; model parameter, (1) mean and (2) std of Gaussian CDF
    %
    % empX
    % vector; empirical numbers of easy distractors
    %
    % empY:
    % vector; empirical proportion choices for easy targets, given a
    % certain number of easy distractors
    %
    % Output:
    % dev:
    % double; loss for parameter combination

    %% Get loss
    pred = 1 - cdf('Normal', empX, par(1), par(2));
    dev = sum((empY - pred).^2, 'omitnan');

end