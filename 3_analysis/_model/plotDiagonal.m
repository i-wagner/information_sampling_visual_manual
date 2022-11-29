function hDiagonal = plotDiagonal(slope)
%PLOTDIAGONAL Plots a diagonal in the current plot

if nargin==0
    slope = 1;
end

setHoldOff = 0;
if ishold == 0
    setHoldOff = 1;
    hold on;
end

hAxis = gca;
x = get(hAxis,'XLim');
y = get(hAxis,'YLim');
if slope==-1
    y = fliplr(y);
end
hDiagonal = plot(x,y,'k--');
set(hDiagonal,'HandleVisibility','off');
%set(hDiagonal,'LineStyle','-','Color',[0.5 0.5 0.5]);
set(hDiagonal,'LineStyle','-'); setappdata(hDiagonal,'PostProcessing','AxisLineWidth');
if setHoldOff == 1
    hold off;
end