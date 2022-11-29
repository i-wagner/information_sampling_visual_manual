function [] = infSampling_generateLUT(inp_setSizes, inp_par_bounds, inp_par_precision, inp_biasedFirstFix)

    % Generates lookup-table for bias/set-size combinations and saves it to
    % drive
    % Input
    % inp_setSizes:       matrix with set sizes that should be considered;
    %                     rows are set-sizes, columns are targets
    % inp_par_bounds:     lower/upper bound of bias
    % inp_par_precision:  number of decimals parameter values are rounded to
    % inp_biasedFirstFix: Bias first fixation (1) or not (0)
    % Output
    % --

    %% Generate bias values
    precision = 10^-inp_par_precision;

    pars = (inp_par_bounds(1):precision:inp_par_bounds(2))';
    pars = repmat(pars, size(inp_setSizes, 1), 1);
    pars = round(pars, inp_par_precision);


    %% Generate set size array
    setSizes = repelem(inp_setSizes, numel(pars)/size(inp_setSizes, 1), 1);


    %% Generate input-part of lookup-table
    lut = [pars setSizes NaN(numel(pars), 5)];
    clear pars setSizes


    %% Generate output-part of lookup-table
    clc

    no_comb = size(lut, 1);
    disp(['Now simulating input-combination # 1 ', 'out of ', num2str(no_comb), ' input-combinations.']);
    for c = 1:no_comb % Combinations of model inputs

        % Progress
        if mod(c, 10^inp_par_precision) == 0

            disp(['Now simulating input-combination # ', num2str(c), ' out of ', num2str(no_comb), ' input-combinations.']);

        end

        % Predict
        [allProb, allFix] = recursiveProb(lut(c, 1), lut(c, 2:3), inp_biasedFirstFix);

        sumChoice    = NaN(1, 2);
        sumFixSet    = NaN(1, 2);
        sumFixChoice = NaN(1, 2);
        for s = 1:2 % Set

            sumChoice(s) = sum(allProb{s});
            sumFixSet(s) = sum([allFix{1, s}.*allProb{1} allFix{2, s}.*allProb{2}]);

        end
        sumFixChoice(1) = sum([allFix{1,1}.*allProb{1} allFix{2,2}.*allProb{2}]);
        sumFixChoice(2) = sum([allFix{1,2}.*allProb{1} allFix{2,1}.*allProb{2}]);

        % Store
        lut(c, 4:8) = [sumChoice, sumFixChoice, sum(sumFixChoice, 2)];

    end
    lut(:, 3) = []; % To save memory, and because it is redundant, get rid of set size difficult


    %% Save lookup-table to drive
    switch inp_biasedFirstFix

        case 1
            save(['infSampling_biasedFirstFix_lut_', num2str(inp_par_precision), '.mat'], 'lut', '-v7.3');

        case 0
            save(['infSampling_unbiasedFirstFix_lut_', num2str(inp_par_precision), '.mat'], 'lut', '-v7.3');

    end

end