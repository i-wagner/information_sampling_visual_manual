function printFigure(fh,size,type,fileName,fileType,subindex,crop)
% parameters:
% 1: figure handle
% 2: position and size in cm [left, bottom, width, height]
% 3: formatting ('paper','poster','presentation','icon')
% 4: filename
% 5: graphic format as in print, i.e '-dpng', '-depsc'
% 6: panel labels ('lcletter', 'letter' or [])
% 7: crop figure (0, 1, 2)

if nargin <6 || isempty(subindex)
    subindex = 0;
end
if nargin<7||isempty(crop)
    crop = 0;
end
% set size
%set(fh,'Color',[1 1 1]);
set(fh,'InvertHardCopy','off');
units = get(fh,'PaperUnits');
set(fh,'PaperUnits','centimeters');
set(fh, 'PaperPositionMode', 'manual');
set(fh, 'PaperSize', size(3:4));
set(fh, 'PaperPosition', size);
set(fh,'PaperUnits',units);
set(fh,'Units','centimeters');
set(fh, 'Position', size);
set(fh,'Units',units);
pause(0.5);
if subindex
    subindex_extended(subindex);
    pause(0.5);
end

% format axes
prepareFigure_acs(fh,type);
figure(fh);

% save file
addpath('E:/MatlabToolbox/export_fig');
set(fh,'Color','w');
if crop==1    
    export_fig (fileName,'-r600','-q101');
elseif crop==2
    export_fig (fileName,'-r600','-q101','-nocrop');    
else
    %export_fig (fileName,'-r600','-q101','-nocrop');
    print(fh,fileType,fileName,'-r600');%'-r1000');%'-r600');
end