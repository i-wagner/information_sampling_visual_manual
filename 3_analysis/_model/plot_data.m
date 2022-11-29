function [] = plot_data(all)
%PLOT_DATA Summary of this function goes here
%   Detailed explanation goes here

% single vs double
LIMS(1,:) = [0 1];
LIMS(2,:) = [0 1];
LIMS(3,:) = [0 1];
LIMS(4,:) = [0 5];
LIMS(5,:) = [0 5];
LIMS(6,:) = [0 5];
LIMS(7,:) = [0 1];
LIMS(8,:) = [0 1];
LIMS(9,:) = [0 1];
LABEL ={'Accuracy easy';'Accuracy diff';'Accuracy';'Non-search time easy';'Non-seach time diff';'Non-sarch time';'Search time easy';'Search time difficult';'Search time';};
figure;
for c=1:size(LIMS,1)
    switch c
        case 1
            x = all.data.single.accuracy(:,1);
            y = all.data.double.accuracy(:,1);
        case 2
            x = all.data.single.accuracy(:,2);
            y = all.data.double.accuracy(:,2);
        case 3
            x = all.data.single.accuracy(:,3);
            y = all.data.double.accuracy(:,3);            
        case 4
            x = all.data.single.non_search_time(:,1);
            y = all.data.double.non_search_time(:,1);
        case 5
            x = all.data.single.non_search_time(:,2);
            y = all.data.double.non_search_time(:,2);
        case 6
            x = all.data.single.non_search_time(:,3);
            y = all.data.double.non_search_time(:,3); 
        case 7
            x = all.data.single.search_time(:,1);
            y = all.data.double.search_time(:,1);
        case 8
            x = all.data.single.search_time(:,2);
            y = all.data.double.search_time(:,2);
        case 9
            x = all.data.single.search_time(:,3);
            y = all.data.double.search_time(:,3);            
    end
    subplot(3,3,c);
    plot(x,y,'ko');
    hold all;
    plotMean(x,y,'k');
    title(LABEL{c});
    xlabel('Single data');
    ylabel('Double data');
    xlim(LIMS(c,:)); set(gca,'XTick',linspace(LIMS(c,1),LIMS(c,2),6));
    squarePlot(gca);
    hold all;  
    plotDiagonal;
    axis square;
end
if all.params.printFigures
    printFigure(gcf,[0 0 30 30],'Paper', 'fig/SingleVsDouble.png','-dpng');
end
end

