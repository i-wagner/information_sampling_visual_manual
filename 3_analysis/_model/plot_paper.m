function [] = plot_paper(all)
%PLOT_PAPER Summary of this function goes here
%   Detailed explanation goes here

figure;
subplot(2,2,1);
x = all.params.set_sizes(:,1);
[ci m] = ci_mean(all.data.double.choices);
% h = errorbar(x,m,ci,'ko-'); set(h(1),'MarkerFaceColor','k','MarkerEdgeColor','w');
for j=1:numel(m)
    h = line([x(j) x(j)],[m(j)-ci(j) m(j)+ci(j)]);set(h,'Color','k');
    set(h,'HandleVisibility','off');
    hold all;
end
h = plot(x,m,'ko-'); set(h(1),'MarkerFaceColor','k','MarkerEdgeColor','w');

for i=1:3
    [ci m] = ci_mean(all.model.choices(:,:,i));
    for j=1:numel(m)
        h = line([x(j) x(j)]+i.*0.1,[m(j)-ci(j) m(j)+ci(j)]);set(h,'Color',all.params.color_model(i));
        set(h,'HandleVisibility','off');
    end
    h = plot(x+i*0.1,m,'ko-'); set(h,'Color',all.params.color_model(i)); set(h(1),'MarkerFaceColor',all.params.color_model(i),'MarkerEdgeColor','w');
    %h = errorbar(x+i*0.1,m,ci,'ko-'); set(h,'Color',all.params.color_model(i)); set(h(1),'MarkerFaceColor',all.params.color_model(i),'MarkerEdgeColor','w');
end
xlim([min(all.params.set_sizes(:,1)) max(all.params.set_sizes(:,1))+0.5]);
set(gca,'XTick',all.params.set_sizes(:,1));
ylim([0 1]);
plotHorizontal(0.5);
plotVertical(all.params.set_size/2);
axis square;
h = legend(['data';all.params.label_model]);
setappdata(h,'LegendLocation','SouthWest');
xlabel('Set size easy target');
ylabel('Proportion choices easy target');

subplot(2,2,2);
data=all.model.weights;
data=data(sum(double(isnan(data)),2)<1,:);
data=sortrows(data,-3);
h = bar(data,'stacked');
for m=1:3
    set(h(m),'EdgeColor','None','FaceColor',all.params.color_model(m));
end
hold all;
axis square;
xlim([0.5 size(data,1)+0.5]);
set(gca,'XTick',1:2:size(data,1));
ylim([0 1]);
xlabel('Participant');
ylabel('Model probability');
for s=1:size(data,1)
    if (data(s,3) > data(s,2))&&(data(s,3) > data(s,1))
        plot(s,0.95,'w*');
    end
end

subplot(2,2,3);
m = 3;
y = all.data.double.perf;
x = all.model.perf(:,m);
sel = ~isnan(x);
x = x(sel);
y = y(sel);
h = plot(x,y,'ko'); set(h,'Color',all.params.color_model(m),'MarkerFaceColor',all.params.color_model(m),'MarkerEdgeColor','w');
hold all;
[b,bint,res,resint,stats] = regress(y,cat(2,ones(size(x)),x));
%text(0.5,2.5,sprintf('y = %.2f+%.2f x; R² = %.2f; F(%d) = %.2f, p = %.3f;\n',b(1),b(2),roundn(stats(1),2),numel(x)-2,roundn(stats(2),2),roundn(stats(3),3)));
% text(0.25,1.4,sprintf('y = %.2f+%.2f x\nR² = %.2f',b(1),b(2),roundn(stats(1),2)));
% [yupp, ylow, xval] = regress_ci(0.05,b,x,y);
% yhat = b(1)+xval.*b(2);
% h = plot(xval,ylow,'k--'); set(h,'Color',all.params.color_model(m));
% h = plot(xval,yupp,'k--'); set(h,'Color',all.params.color_model(m));
% h = plot(xval,yhat,'k-'); set(h,'Color',all.params.color_model(m));

xlim([0 1.5]); ylim([0 1.5]);
plotDiagonal;
axis square;
% title(all.params.label_model{m});
xlabel('Prediction');
ylabel('Empirical');

subplot(2,2,4);
m=3;
% x=linspace(0,2,10);
% n_weights = hist(all.model.noise(:,m),x);
x = linspace(0,1,20);
n_weights = hist(all.model.perf(:,m)./all.model.perf_perfect(:,m),x);
h = bar(x,n_weights);
hold all;
% xlim([-0.05 2.05]);
xlim([0.45 1.05]);
ylim([0 10]);
axis square;
set(h,'EdgeColor','None','FaceColor',all.params.color_model(m));
% mean_weight = nanmean(all.model.noise(:,m));
mean_weight = nanmean(all.model.perf(:,m)./all.model.perf_perfect(:,m));
hp = plotVertical(mean_weight);set(hp,'Color',all.params.color_model(m));
% xlabel('SD of decision noise');
xlabel('Efficiency');
ylabel('N');


if all.params.printFigures
    printFigure(gcf,[0 0 20 20],'Paper', 'fig/ModelResults.png','-dpng');
end
end

