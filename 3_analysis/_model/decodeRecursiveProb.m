function [sumChoice, sumFixChoice, sumFixSet] = decodeRecursiveProb(bias, setSize, BIASFIRSTFIX)

    %% Predict proportion choices easy target, # fixations easy/difficult set, # fixations chosen set
    [allProb, allFix] = recursiveProb(bias, setSize, BIASFIRSTFIX);


    %% Output
    for s = 1:2 % Set

        sumChoice(s) = sum(allProb{s});
        sumFixSet(s) = sum([allFix{1, s} .* allProb{1} allFix{2, s} .* allProb{2}]);

    end
    sumFixChoice(1) = sum([allFix{1, 1} .* allProb{1} allFix{2,2} .* allProb{2}]);
    sumFixChoice(2) = sum([allFix{1, 2} .* allProb{1} allFix{2,1} .* allProb{2}]);

end