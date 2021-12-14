function [all] = read_data(all)
%READ_DATA Summary of this function goes here
%   Detailed explanation goes here
single_data = dlmread('dataSingleTarget.txt');
double_data = dlmread('dataDoubleTarget.txt');
[~, column_data] = xlsread('labels.xlsx');
all.params.n = size(single_data,1);

all.data.single.accuracy(:,1) = single_data(:,3);
all.data.single.accuracy(:,2) = single_data(:,4);
all.data.single.accuracy(:,3) = mean(all.data.single.accuracy(:,[1 2]),2);
all.data.single.n_fix(:,:,1) = single_data(:,[29:37]);
all.data.single.n_fix(:,:,2) = single_data(:,[56:64]);
all.data.single.non_search_time(:,1) = single_data(:,70)./1000;
all.data.single.non_search_time(:,2) = single_data(:,74)./1000;
all.data.single.non_search_time(:,3) = single_data(:,66)./1000;
all.data.single.search_time(:,1) = single_data(:,69)./1000;
all.data.single.search_time(:,2) = single_data(:,73)./1000;
all.data.single.search_time(:,3) = single_data(:,65)./1000;

if ~all.params.use_empirical_fix_num
    for i=1:size(all.params.set_sizes,1)
        all.data.pred.mean_item_per_set(1,i,1) = (all.params.set_sizes(i,1)-1)/2;
        all.data.pred.mean_item_per_set(1,i,2) = (all.params.set_sizes(i,2)-1)/2;
        all.data.pred.mean_item_per_set(1,i,3) = mean(all.data.pred.mean_item_per_set(1,i,[1 2]));
    end
    all.data.pred.mean_item_per_set = repmat(all.data.pred.mean_item_per_set,[all.params.n 1 1]);
else
    for i=1:size(all.params.set_sizes,1)
        all.data.pred.mean_item_per_set(s,i,1) = all.data.single.n_fix(:,i,1);
        all.data.pred.mean_item_per_set(s,i,2) = all.data.single.n_fix(:,size(all.params.set_sizes,1)+1-i,2);
        all.data.pred.mean_item_per_set(s,i,3) = mean(all.data.pred.mean_item_per_set(s,i,[1 2]));
    end
end

all.data.double.accuracy(:,1) = double_data(:,3);
all.data.double.accuracy(:,2) = double_data(:,4);
all.data.double.accuracy(:,3) = mean(all.data.double.accuracy(:,[1 2]),2);
all.data.double.n_fix(:,:,1) = double_data(:,[29:37]);
all.data.double.n_fix(:,:,2) = double_data(:,[56:64]);
all.data.double.non_search_time(:,1) = double_data(:,70)./1000;
all.data.double.non_search_time(:,2) = double_data(:,74)./1000;
all.data.double.non_search_time(:,3) = double_data(:,66)./1000;
all.data.double.search_time(:,1) = double_data(:,69)./1000;
all.data.double.search_time(:,2) = double_data(:,73)./1000;
all.data.double.search_time(:,3) = double_data(:,65)./1000;

if all.params.use_single_pred
    all.data.pred.accuracy = all.data.single.accuracy;
    all.data.pred.non_search_time = all.data.single.non_search_time;
    all.data.pred.search_time = all.data.single.search_time;
else
    all.data.pred.accuracy = all.data.double.accuracy;
    all.data.pred.non_search_time = all.data.double.non_search_time;
    all.data.pred.search_time = all.data.double.search_time;
end
all.data.pred.accuracy(all.data.pred.accuracy<0.5) = 0.5;


for ne = 1:9
    all.data.double.choices(:,ne) = double_data(:,78+ne);        
end
all.data.double.perf = ((double_data(:,2).*double_data(:,97).*all.params.payoff(1,1))+((1-double_data(:,2)).*double_data(:,97).*all.params.payoff(2,1)))/(all.params.time*60);
all.data.single.perf = ((single_data(:,2).*single_data(:,97).*all.params.payoff(1,1))+((1-single_data(:,2)).*single_data(:,97).*all.params.payoff(2,1)))/(all.params.time*60);
end

