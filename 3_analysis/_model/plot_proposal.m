function [] = plot_proposal(all)
%PLOT_PAPER Summary of this function goes here
%   Detailed explanation goes here

COLOR = [241,163,64; 153,142,195]./255;
figure;
subplot(1,2,1);
x = all.params.set_sizes(:,1);
[ci m] = ci_mean(all.data.double.choices);
for j=1:numel(m)
    h = line([x(j) x(j)],[m(j)-ci(j) m(j)+ci(j)]);set(h,'Color',COLOR(1,:));
    set(h,'HandleVisibility','off');
    hold all;
end
h = plot(x,m,'ko-');  set(h,'Color',COLOR(1,:)); set(h(1),'MarkerFaceColor',COLOR(1,:),'MarkerEdgeColor','w');

i = 3;
[ci m] = ci_mean(all.model.choices(:,:,i));
for j=1:numel(m)
    h = line([x(j) x(j)]+0.1,[m(j)-ci(j) m(j)+ci(j)]);set(h,'Color',COLOR(2,:));
    set(h,'HandleVisibility','off');
end
h = plot(x+0.1,m,'ko-'); set(h,'Color',COLOR(2,:)); set(h(1),'MarkerFaceColor',COLOR(2,:),'MarkerEdgeColor','w');

xlim([min(all.params.set_sizes(:,1)) max(all.params.set_sizes(:,1))+0.5]);
set(gca,'XTick',all.params.set_sizes(:,1));
ylim([0 1]);
plotHorizontal(0.5);
plotVertical(all.params.set_size/2);
axis square;
h = legend('Data','Model');
setappdata(h,'LegendLocation','SouthWest');
xlabel('Set size easy target');
ylabel('Proportion choices easy target');


subplot(1,2,2);
m = 3;
y = all.data.double.perf;
x = all.model.perf(:,m);
sel = ~isnan(x);
x = x(sel);
y = y(sel);
h = plot(x,y,'ko'); set(h,'Color','k','MarkerFaceColor','k','MarkerEdgeColor','w');
hold all;
[b,bint,res,resint,stats] = regress(y,cat(2,ones(size(x)),x));
%text(0.5,2.5,sprintf('y = %.2f+%.2f x; R² = %.2f; F(%d) = %.2f, p = %.3f;\n',b(1),b(2),roundn(stats(1),2),numel(x)-2,roundn(stats(2),2),roundn(stats(3),3)));
% text(0.25,1.4,sprintf('y = %.2f+%.2f x\nR² = %.2f',b(1),b(2),roundn(stats(1),2)));
% [yupp, ylow, xval] = regress_ci(0.05,b,x,y);
% yhat = b(1)+xval.*b(2);
% h = plot(xval,ylow,'k--'); set(h,'Color','k');
% h = plot(xval,yupp,'k--'); set(h,'Color','k');
% h = plot(xval,yhat,'k-'); set(h,'Color','k');

xlim([0 1.5]); ylim([0 1.5]);
plotDiagonal;
axis square;
% title(all.params.label_model{m});
xlabel('Model [gain in cents/s]');
ylabel('Data [gain in cents/s]');
set(gca,'YColor',COLOR(1,:));
set(gca,'XColor',COLOR(2,:));

if all.params.printFigures
    printFigure(gcf,[0 0 14 7],'Paper', 'fig/ModelResults.png','-dpng');
end
end

