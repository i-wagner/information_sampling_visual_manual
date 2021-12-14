function [all] = fit_regression(all)
%FIT_REGRESSION Summary of this function goes here
%   Detailed explanation goes here
for s=1:all.params.n
    x = all.params.set_sizes(:,1);
    y = all.data.double.choices(s,:);
    
    if sum(isnan(y))==0
        [all.reg.fit(s,:) all.reg.fitInt(s,:,:)] = regress(y'-0.5,[ones(size(x)) x-5]); %subtraction necessary to get interpretable parameters
        all.reg.xn(s,:) = x-5;
        all.reg.yn(s,:) = all.reg.fit(s,1)+all.reg.fit(s,2).*all.reg.xn(s,:);
        all.reg.xn(s,:) = all.reg.xn(s,:)+5;
        all.reg.yn(s,:) = all.reg.yn(s,:)+0.5;
    else
        all.reg.yn(s,1:numel(x)) = NaN;
        all.reg.xn(s,1:numel(x)) = NaN;
        all.reg.fit(s,1:2) = NaN;
        all.reg.fitInt(s,1:2,1:2) = NaN;
    end
end

end

