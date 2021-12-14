function h = plotVertical(interception,axisLineWidth,dashed)
%PLOTDIAGONAL Plots a diagonal in the current plot

if nargin<2
    axisLineWidth = 1;
end
if nargin<3
    dashed = 0;
end

setHoldOff = 0;
if ishold == 0
    setHoldOff = 1;
    hold on;
end

hAxis = gca;
y = get(hAxis,'YLim');
if ~dashed
    h = plot([interception interception],y,'k--');
    %set(h,'LineStyle','-','Color',[0.5 0.5 0.5]);
    set(h,'LineStyle','-');    
else
    h = dashline([interception interception],y,dashed,dashed,dashed,dashed);
end
set(h,'HandleVisibility','off');
if axisLineWidth
    setappdata(h,'PostProcessing','AxisLineWidth');
end
if setHoldOff == 1
    hold off;
end