function [all] = get_params(all)
%GET_PARAMS Summary of this function goes here
%   Detailed explanation goes here

% settings
all.params.printFigures = 0;
all.params.use_empirical_fix_num = 0;
all.params.use_single_pred = 1;

% figures
all.params.color_difficulty = 'br';
all.params.marker_difficulty = '^v';
all.params.color_model = 'gbc';
all.params.label_model = {'difficulty';'set size';'both'};

% paradigm
all.params.time = 6.5;
all.params.set_size = 10;
all.params.set_sizes(:,1) = 1:(all.params.set_size-1);
all.params.set_sizes(:,2) = all.params.set_size-all.params.set_sizes(:,1);
all.params.payoff = [2 2; -2 -2];

% model
all.params.optionsMinsearch = optimset('MaxFunEvals',5000, 'MaxIter',5000, 'TolFun',1e-12, 'TolX',1e-12);
all.params.noise.init = 0.1;
all.params.noise.min = 0;
all.params.noise.max = 2;

end

