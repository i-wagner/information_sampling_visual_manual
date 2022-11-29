function [choice] = predict_choices(sd,predictor)
%PREDICT_CHOICES Summary of this function goes here
%   Detailed explanation goes here

    N = 100000;
    if all(isnan(predictor))

        choice = NaN(1, numel(predictor));

    else

        % Fitting can be sped up if the above calculations are condensed into
        % one line of code (less variables have to allocated, I guess?) and if
        % the predictor is not replicated (less memory demands). This has a
        % neglible effect when N is small (e.g., 10k), but has quite the impact
        % on processing time when N is large (e.g., 100k)
        choice = mean(double(predictor + repmat(randn(N,1).*sd,[1 numel(predictor)])<0), 'omitnan');

    end

end