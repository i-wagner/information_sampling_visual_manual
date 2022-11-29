function  squarePlot( h )

set(h,'YLim',get(h,'XLim'));
if ~strcmp(get(h,'XScale'),'log')
    set(h,'YTick',get(h,'XTick'));
    set(h,'YTickLabel',get(h,'XTickLabel'));
    set(h,'YMinorTick',get(h,'XMinorTick'));    
end
set(h,'YScale',get(h,'XScale'));
set(h, 'PlotBoxAspectRatio',[1 1 1], 'DataAspectRatioMode','auto');
