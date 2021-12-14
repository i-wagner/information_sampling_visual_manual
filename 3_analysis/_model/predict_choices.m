function [choice] = predict_choices(sd,predictor)
%PREDICT_CHOICES Summary of this function goes here
%   Detailed explanation goes here
N = 10000;
noise = randn(N,1).*sd;
noise = repmat(noise,[1 numel(predictor)]);
predictor = repmat(predictor,[N 1]);
criterion = predictor+noise;
decision = double(criterion<0);
choice = nanmean(decision);
if all(isnan(criterion(:)))
    choice = NaN(1, numel(choice));
end

end

