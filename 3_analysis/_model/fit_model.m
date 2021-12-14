function [all] = fit_model(all)
%FIT_MODEL Summary of this function goes here
%   Detailed explanation goes here

for s=1:all.params.n
    for m=1:3        
        % generate predictors
        for i=1:2
            switch m
                case 1 % difficulty
                    all.model.gain(s,:,i,m) = ...
                        (all.data.pred.accuracy(s,i).*all.params.payoff(1,i)...
                        +(1-all.data.pred.accuracy(s,i)).*all.params.payoff(2,i))...
                        ./(all.data.pred.mean_item_per_set(s,:,3).*all.data.pred.search_time(s,i)...
                        +all.data.single.non_search_time(s,i));
                case 2 % set size
                    all.model.gain(s,:,i,m) = ...
                        (all.data.pred.accuracy(s,3).*all.params.payoff(1,i)...
                        +(1-all.data.pred.accuracy(s,3)).*all.params.payoff(2,i))...
                        ./(all.data.pred.mean_item_per_set(s,:,i).*all.data.pred.search_time(s,3)...
                        +all.data.single.non_search_time(s,3));
                case 3 % difficulty & set size
                    all.model.gain(s,:,i,m) = ...
                        (all.data.pred.accuracy(s,i).*all.params.payoff(1,i)...
                        +(1-all.data.pred.accuracy(s,i)).*all.params.payoff(2,i))...
                        ./(all.data.pred.mean_item_per_set(s,:,i).*all.data.pred.search_time(s,i)...
                        +all.data.single.non_search_time(s,i));     
%                 case 4 % difficulty & set size without search duration
%                     all.model.gain(s,:,i,m) = ...
%                         (all.data.pred.accuracy(s,i).*all.params.set_sizes(:,i).*all.params.payoff(1,i));                    
            end
        end
        
        % fit noise and predict choices
        all.model.relative_gain(s,:,m) = all.model.gain(s,:,2,m)-all.model.gain(s,:,1,m);
        all.model.noise(s,m) = fminsearchbnd(@predict_choices_dev,all.params.noise.init,all.params.noise.min,all.params.noise.max,all.params.optionsMinsearch,all.data.double.choices(s,:),squeeze(all.model.relative_gain(s,:,m)));
        all.model.choices(s,:,m) = predict_choices(all.model.noise(s,m),squeeze(all.model.relative_gain(s,:,m)));      
        all.model.rss(s,m) = sum((squeeze(all.data.double.choices(s,:))-squeeze(all.model.choices(s,:,m))).^2);  
        all.model.choices_perfect(s,:,m) = predict_choices(0,squeeze(all.model.relative_gain(s,:,m)));
        sel = find(all.model.relative_gain(s,:,m)==0);
        all.model.choices_perfect(s,sel,m) = 0.5;
        all.model.perf(s,m) = mean(squeeze(all.model.choices(s,:,m)).*squeeze(all.model.gain(s,:,1,m))+(1-squeeze(all.model.choices(s,:,m))).*squeeze(all.model.gain(s,:,2,m)));
        all.model.perf_perfect(s,m) = mean(squeeze(all.model.choices_perfect(s,:,m)).*squeeze(all.model.gain(s,:,1,m))+(1-squeeze(all.model.choices_perfect(s,:,m))).*squeeze(all.model.gain(s,:,2,m)));
        
    end
    
    % calculate model weights
    [all.model.aic(s,:) all.model.bic(s,:)] = informationCriterion(all.model.rss(s,:),1,9);
    all.model.weights(s,:) = informationWeights(all.model.aic(s,:));
end


