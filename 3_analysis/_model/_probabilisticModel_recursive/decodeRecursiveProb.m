function [sumChoice sumFixChoice sumFixSet] = decodeRecursiveProb(bias,setSize)
[allProb allFix] = recursiveProb(bias,setSize);
for s=1:2
    sumChoice(s)  = sum(allProb{s});
    sumFixSet(s) = sum([allFix{1,s}.*allProb{1} allFix{2,s}.*allProb{2}]);
end
sumFixChoice(1) = sum([allFix{1,1}.*allProb{1} allFix{2,2}.*allProb{2}]);
sumFixChoice(2) = sum([allFix{1,2}.*allProb{1} allFix{2,1}.*allProb{2}]);

