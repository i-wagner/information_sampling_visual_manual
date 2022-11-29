function [] = plot_presentation(all)

color_model = [[215  94 110 ]./255; ...
               [250 156 156]./255; ...
               [250 204 221]./255];

close all
fig.h = figure;
subplot(1,2,1);
x = all.params.set_sizes(:,1)-1;
[ci m] = ci_mean(all.data.double.choices);
% h = errorbar(x,m,ci,'ko-'); set(h(1),'MarkerFaceColor','k','MarkerEdgeColor','w');
for j=1:numel(m)
    h = line([x(j) x(j)],[m(j)-ci(j) m(j)+ci(j)]);set(h,'Color','k','LineWidth',2);
    set(h,'HandleVisibility','off');
    hold all;
end
h = plot(x,m,'ko-'); set(h(1),'MarkerFaceColor','k','MarkerEdgeColor','w','MarkerSize',plt.size.mrk_mean,'LineWidth',2);

for i=1:3
    [ci m] = ci_mean(all.model.choices(:,:,i));
    for j=1:numel(m)
        h = line([x(j) x(j)]+i.*0.1,[m(j)-ci(j) m(j)+ci(j)]);set(h,'Color',color_model(i,:),'LineWidth',2);
        set(h,'HandleVisibility','off');
    end
    h = plot(x+i*0.1,m,'ko-'); set(h,'Color',color_model(i,:)); set(h(1),'MarkerFaceColor',color_model(i,:),'MarkerEdgeColor','w','MarkerSize',plt.size.mrk_mean,'LineWidth',2);
    %h = errorbar(x+i*0.1,m,ci,'ko-'); set(h,'Color',all.params.color_model(i)); set(h(1),'MarkerFaceColor',all.params.color_model(i),'MarkerEdgeColor','w');
end
xlim([min(all.params.set_sizes(:,1))-2 max(all.params.set_sizes(:,1))]);
set(gca,'XTick',all.params.set_sizes(:,1)-1);
ylim([0 1]);
yticks(0:0.25:1)
clear h_l; h_l(1) = plotHorizontal(0.5);
set(h_l(1), 'LineStyle', '--', 'LineWidth', 2, 'Color', plt.color.c1)
h_l(2) = plotVertical(4);
set(h_l(2), 'LineStyle', '--', 'LineWidth', 2, 'Color', plt.color.c1)
uistack(h_l, 'bottom')
axis square;
h = legend(['data';all.params.label_model],'Location','SouthWest');
legend box off
% setappdata(h,'LegendLocation','SouthWest');
xlabel('# easy distractors');
ylabel('Proportion choices easy target');

subplot(1,2,2);
data=all.model.weights;
data=data(sum(double(isnan(data)),2)<1,:);
data=sortrows(data,-3);
h = bar(data,'stacked');
for m=1:3
    set(h(m),'EdgeColor','None','FaceColor',color_model(m,:));
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
        plot(s,0.95,'k*','MarkerSize',15);
    end
end
box off

if all.params.printFigures
    opt.imgname = 'fig/ModelResults_presentation.png';
    opt.size    = [50 20];
    opt.save    = 1;
    prepareFigure(fig.h, opt)
%     close; clear c fig opt
%     printFigure_acs(gcf,[0 0 20 20],'Paper', 'fig/ModelResults_presentation.png','-dpng');
end
end

