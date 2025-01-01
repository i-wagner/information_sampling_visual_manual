function [ se m ] = se_mean( raw )

    %SE_MEAN Summary of this function goes here
    %   Detailed explanation goes here

    m = nanmean(raw);
    sd = nanstd(raw);
    %n = length(find(isnan(raw)==0));
    if size(raw,1)==1
        n = length(find(isnan(raw)==0));
    else
        n = sum(~isnan(raw)); % changed 16.12.14
        %n = length(find(isnan(raw(:,1,1))==0));
    end
    se = sd ./ sqrt(n);
    %se = sqrt(nansum((raw-m).^2) ./ (n .* (n - 1)));

end