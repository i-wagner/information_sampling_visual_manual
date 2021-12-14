function plot_regression(all)
%PLOT_REGRESSION Summary of this function goes here
%   Detailed explanation goes here

figure;
for s=1:all.params.n
    subplot(5,5,s);
    x = all.params.set_sizes(:,1);
    y = all.data.double.choices(s,:);
    xn = all.reg.xn(s,:);
    yn = all.reg.yn(s,:);
    plot(x,y,'ko');
    hold all;
    plot(xn,yn,'k-');
    axis square;
    xlim([min(x)-0.5 max(x)+0.5]);
    ylim([0 1]);
    set(gca,'XTick',x);
    set(gca,'YTick',[0 0.5 1]);
    if s==1
        xlabel('Set size easy');
        ylabel('Proportion easy');
    else
        set(gca,'XTickLabel',[]);
        set(gca,'YTickLabel',[]);
    end
    title(num2str(s));
    plotHorizontal(0.5);
    plotVertical(all.params.set_size/2);    
end
subplot(5,5,all.params.n+1);
plot(all.reg.fit(:,1),all.reg.fit(:,2),'ko');
[ci m h] = plotMean(all.reg.fit(:,1),all.reg.fit(:,2),'k');
delete(h(4));
xlim([-0.5 0.5]);
ylim([-0.2 0.1]);
plotHorizontal(0);
plotVertical(0);
axis square;
xlabel('Intercept (difficulty)');
ylabel('Slope (set size');
if all.params.printFigures
    printFigure(gcf,[0 0 50 50],'Paper','fig/Regression.png','-dpng');
end

end

