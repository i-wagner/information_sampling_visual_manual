function dev = lossCdf(par, empX, empY)

    % Calculates loss when fitting Gaussian CDF to choice data
    %
    % NOTE:
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
    % For our default case, we expect the sigmoid to be negative, i.e., it
    % should have it's highest value at low values of x, and decrease with
    % increasing values of x. For negative values of the SD, we assume a
    % "regular" sgmoid, which has it's highest value at high x
    if sign(par(2)) == -1
        % cdf() needs a positive SD, otherwise it returns NaN
        pred = cdf('Normal', empX, par(1), abs(par(2)));
    else
        pred = 1 - cdf('Normal', empX, par(1), par(2));
    end
    dev = sum((empY - pred).^2, 'omitnan');

end