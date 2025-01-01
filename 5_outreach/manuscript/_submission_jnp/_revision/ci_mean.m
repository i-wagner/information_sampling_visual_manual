function [ ci m ] = ci_mean( raw, errorProb)

    %SE_MEAN Summary of this function goes here
    %   Detailed explanation goes here
    if nargin==1
       errorProb = 0.05;
    end
    [se m] = se_mean(raw);
    %df = size(raw,1)-1;
    df = sum(~isnan(raw))-1; %changed on 29.04.2015
    ci = se .* tinv(1-(errorProb*0.5),df);

end