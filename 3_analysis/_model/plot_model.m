function [] = plot_model(all)
%PLOT_MODEL Summary of this function goes here
%   Detailed explanation goes here

close all;

% plot relationship between model fit and difference in performance
figure;
[secondWeight secondIndex] = max(all.model.weights(:,[1 2]),[],2);
x = secondWeight-all.model.weights(:,3);
y = all.model.perf(:,secondIndex)-all.model.perf(:,3);
plot(x,y,'ko');

% plot relationship between noise and loss of gain
figure;
subplot(1,2,1);
x = all.model.noise(:,3);
y = all.model.perf(:,3)./all.model.perf_perfect(:,3);
%plot(x,y,'ko');
hist(y);
subplot(1,2,2);
y = all.model.weights(:,3);
plot(x,y,'ko');

% plot individual subjects
for s=1:all.params.n
    for m=1:3
        figure(s);
        subplot(3,3,m);
        for i=1:2
            h = plot(all.params.set_sizes(:,1),all.model.gain(s,:,i,m),'ro-'); set(h,'Color',all.params.color_difficulty(i),'Marker',all.params.marker_difficulty(i));
            hold all;
        end
        xlabel('Set size easy');
        ylabel('Individual value [cent/s]');
        axis square;
        xlim([min(all.params.set_sizes(:,1)) max(all.params.set_sizes(:,1))]);
        ylim([0 2]);
        title(all.params.label_model{m});
        
        subplot(3,3,3+m);
        h = plot(all.params.set_sizes(:,1),all.model.relative_gain(s,:,m),'mo-');
        hold all;
        xlabel('Set size easy');
        ylabel('Relative value [cent/s]');
        axis square;
        xlim([min(all.params.set_sizes(:,1)) max(all.params.set_sizes(:,1))]);
        ylim([-1 1]);
        plotHorizontal(0);
        plotVertical(all.params.set_size/2);
        
        subplot(3,3,6+m);
        h = plot(all.params.set_sizes(:,1),all.data.double.choices(s,:),'kd-');
        hold all;       
        h = plot(all.params.set_sizes(:,1),all.model.choices(s,:,m),'ms-');
        h = plot(all.params.set_sizes(:,1),all.model.choices_perfect(s,:,m),'mo--');
        xlabel('Set size easy');
        ylabel('Choices easy');
        axis square;
        xlim([min(all.params.set_sizes(:,1)) max(all.params.set_sizes(:,1))]);
        ylim([0 1]);
        plotHorizontal(0.5);
        plotVertical(all.params.set_size/2);
        title(sprintf('SD of decision noise: %.2f',all.model.noise(s,m)));
    end
    if all.params.printFigures
        printFigure(gcf,[0 0 30 30],'Paper',sprintf('fig/Subject_%d.png',s),'-dpng');
    end
end

% plot model weights
figure;
subplot(1,3,1);
x=linspace(0,1,20);
n_weights = hist(all.model.weights,x);
h = bar(x,n_weights);
hold all;
xlim([-0.05 1.05]);
ylim([0 10]);
for m=1:3
    set(h(m),'EdgeColor','None','FaceColor',all.params.color_model(m));
    mean_weight = nanmean(all.model.weights(:,m));
    hp = plotVertical(mean_weight);set(hp,'Color',all.params.color_model(m));
end

legend(all.params.label_model);
axis square;
xlabel('Probability');
ylabel('N');

subplot(1,3,2);
plot(all.model.weights(:,1),all.model.weights(:,3),'ko');
plotMean(all.model.weights(:,1),all.model.weights(:,3),'k');
xlim([0 1]); set(gca,'XTick',0:0.2:1);
squarePlot(gca);
hold all;
plotDiagonal;
xlabel(sprintf('Probability %s',all.params.label_model{1}));
ylabel(sprintf('Probability %s',all.params.label_model{3}));

subplot(1,3,3);
plot(all.model.weights(:,2),all.model.weights(:,3),'ko');
plotMean(all.model.weights(:,2),all.model.weights(:,3),'k');
xlim([0 1]); set(gca,'XTick',0:0.2:1);
squarePlot(gca);
hold all;
plotDiagonal;
xlabel(sprintf('Probability %s',all.params.label_model{2}));
ylabel(sprintf('Probability %s',all.params.label_model{3}));
if all.params.printFigures
    printFigure(gcf,[0 0 30 10],'Paper','fig/Model_weights.png','-dpng');
end

% decision noise
figure;
subplot(1,2,1);
x=linspace(0,2,20);
n_weights = hist(all.model.noise,x);
h = bar(x,n_weights);
hold all;
xlim([-0.05 2.05]);
ylim([0 10]);
for m=1:3
    set(h(m),'EdgeColor','None','FaceColor',all.params.color_model(m));
    mean_weight = nanmean(all.model.noise(:,m));
    hp = plotVertical(mean_weight);set(hp,'Color',all.params.color_model(m));
end
legend(all.params.label_model);
axis square;
xlabel('Decision noise [cents/s]');
ylabel('N');

subplot(1,2,2);
x=linspace(0,1,20);
y=all.model.perf./all.model.perf_perfect;
n_weights = hist(y,x);
h = bar(x,n_weights);
hold all;
xlim([0.45 1.05]);
ylim([0 10]);
for m=1:3
    set(h(m),'EdgeColor','None','FaceColor',all.params.color_model(m));
    mean_weight = nanmean(y(:,m));
    hp = plotVertical(mean_weight);set(hp,'Color',all.params.color_model(m));
end
legend(all.params.label_model);
axis square;
xlabel('Efficiency');
ylabel('N');

% for m=1:3
%     subplot(2,2,m+1);
%     x = all.model.perf(:,m);
%     y = all.model.perf_perfect(:,m);
%     h = plot(x,y,'ko'); set(h,'Color',all.params.color_model(m));
%     hold all;
%     plotMean(x,y,all.params.color_model(m));
%     title(all.params.label_model{m});
%     xlim([0 1.5]);
%     squarePlot(gca);
%     plotDiagonal;
% end
if all.params.printFigures
    printFigure(gcf,[0 0 20 10],'Paper','fig/Model_noise.png','-dpng');
end

% efficiency
figure;
color = {[0.5 0.5 0.5]; [0 0 0]};
for m=1:3
    subplot(1,3,m);
    perfect = all.model.perf_perfect(:,m);
    noise = all.model.perf(:,m);
    emp = all.data.double.perf;
    noise = noise./perfect;
    emp = emp./perfect;
    data = cat(2,noise,emp);
    x=linspace(0,1.5,20);
    n = hist(data,x);
    h = bar(x,n);
    hold all;
    xlim([-0.05 1.55]);
    ylim([0 10]);
    for i=1:2
        set(h(i),'EdgeColor','None','FaceColor',color{i});
        mean_weight = nanmean(data(:,i));
        hp = plotVertical(mean_weight);set(hp,'Color',color{i});
    end
    legend('Decision noise','Empirical');
    axis square;
    xlabel('Efficiency');
    ylabel('N');
end
if all.params.printFigures
    printFigure(gcf,[0 0 30 10],'Paper','fig/Model_efficiency.png','-dpng');
end

% performance
figure;
for m=1:3
    for t=1:2
        subplot(2,3,(t-1)*3+m);
        y = all.data.double.perf;
        if t==1
            x = all.model.perf(:,m);
        else
            x = all.model.perf_perfect(:,m);
        end
        h = plot(x,y,'ko'); set(h,'Color',all.params.color_model(m));
        hold all;
        [b,bint,res,resint,stats] = regress(y,cat(2,ones(size(x)),x));
        %text(0.5,2.5,sprintf('y = %.2f+%.2f x; R² = %.2f; F(%d) = %.2f, p = %.3f;\n',b(1),b(2),roundn(stats(1),2),numel(x)-2,roundn(stats(2),2),roundn(stats(3),3)));
        text(0.25,1.4,sprintf('y = %.2f+%.2f x\nR² = %.2f',b(1),b(2),round(stats(1),2)));
%         [yupp, ylow, xval] = regress_ci(0.05,b,x,y);
%         yhat = b(1)+xval.*b(2);
%         h = plot(xval,ylow,'k--'); set(h,'Color',all.params.color_model(m));
%         h = plot(xval,yupp,'k--'); set(h,'Color',all.params.color_model(m));
%         h = plot(xval,yhat,'k-'); set(h,'Color',all.params.color_model(m));
        
        xlim([0 1.5]); ylim([0 1.5]);
        plotDiagonal;
        axis square;
        title(all.params.label_model{m});
        xlabel('Prediction');
        ylabel('Empirical');
    end
end
if all.params.printFigures
    printFigure(gcf,[0 0 30 20],'Paper', 'fig/PredictionPerformance.png','-dpng');
end
end

