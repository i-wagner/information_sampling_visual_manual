function [allProb allFix] = recursiveProb(bias,setSize,remainingProb,fixCount,allProb,allFix)
%RECURSIVEMODEL Summary of this function goes here
%   Detailed explanation goes here

UNBIASED_FIRST_FIXATION = 1;
% initialize starting values
if nargin<6
    remainingProb = 1;
    fixCount = [0 0];   
    for s=1:2
        allProb{s} = [];
        for f=1:2
            allFix{s,f} = [];
        end
    end
end
for s=1:2
    % probability to select set
    if UNBIASED_FIRST_FIXATION && sum(fixCount) == 0
        sel(s) = setSize(s)./(setSize(1)+setSize(2));
    else
        if bias<1
            if s==1
                sel(s) = setSize(s)./(setSize(1)+(bias.*setSize(2)));
            else
                sel(s) = (bias.*setSize(s))./(setSize(1)+(bias.*setSize(2)));
            end
        else
            if s==1
                sel(s) = ((2-bias).*setSize(s))./((2-bias).*setSize(1)+setSize(2));
            else
                sel(s) = setSize(s)./((2-bias).*setSize(1)+setSize(2));
            end
        end
    end
    % probability to find target in set
    hitProbability(s) = sel(s).*(1/setSize(s)).*remainingProb;
    % probability to miss target in set
    missProbability(s) = sel(s).*((setSize(s)-1)./setSize(s)).*remainingProb;
    
    % store finished path when target has been found
    allProb{s} = [allProb{s} hitProbability(s)];
    allFix{s,s} = [allFix{s,s} fixCount(s)+1];
    allFix{s,3-s} = [allFix{s,3-s} fixCount(3-s)];
    
    % continue path if more than one element is left in set
    if setSize(s)>1
        tmp_setSize = setSize;
        tmp_setSize(s) = tmp_setSize(s)-1;
        tmp_fix = fixCount;
        tmp_fix(s) = tmp_fix(s)+1;
        [allProb allFix] = recursiveProb(bias,tmp_setSize,missProbability(s),tmp_fix,allProb,allFix);
    end
end