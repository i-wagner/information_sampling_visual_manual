function prepareFigure_acs(fh,type)

% Special appdata settings:
% LegendLocation: see location values of legend
% PostProcessing: off, Psychometric Function, PsychometricData, 
% FontTemplate

%% Settings
type = lower(type);
AX_FN = 'Helvetica';
LB_FN = 'Helvetica';
AX_FA = 'Normal';
LB_FA = 'Normal';
if strcmp(type,'poster')==1
    AX_FS = 24;%20;%16;
    AX_FW = 'Normal';
    LB_FS = 24;%32;
    LB_FW = 'Normal';
    AX_LW = 2;
    PL_LW = 2;
    PL_MS = 20;%16;
elseif strcmp(type,'paper')==1
    AX_FS = 9;
    AX_FW = 'Normal';
    LB_FS = 9;
    LB_FW = 'Normal';
    AX_LW = 0.5;
    PL_LW = 1;
    PL_MS = 6;
    AR_FS = 5;
elseif strcmp(type,'grant')==1
    AX_FN = 'Arial';
    LB_FN = 'Arial';    
    AX_FS = 9;
    AX_FW = 'Normal';
    LB_FS = 9;
    LB_FW = 'Normal';
    AX_LW = 0.5;
    PL_LW = 1;
    PL_MS = 6;
    AR_FS = 5;    
elseif strcmp(type,'presentation')==1
    AX_FS = 16;
    AX_FW = 'Normal';
    LB_FS = 24;
    LB_FW = 'Normal';
    AX_LW = 2;
    PL_LW = 2;
    PL_MS = 16;
elseif strcmp(type,'tony')==1
    AX_FS = 9;
    AX_FW = 'Normal';
    %AX_FA = 'Oblique';
    LB_FS = 9;
    LB_FW = 'Normal';
    ID_FS = 12;
    %LB_FA = 'Oblique';
    AX_LW = 0.5;%0.5;
    PL_LW = 0.5;%1;
    PL_MS = 6;
    AR_FS = 5;
elseif strcmp(type,'icon')==1
    AX_FS = 9;
    AX_FW = 'Normal';
    LB_FS = 9;
    LB_FW = 'Normal';
    AX_LW = 0.5;
    PL_LW = 1;
    PL_MS = 6;
    AR_FS = 5;
elseif strcmp(type,'pnas')==1
    AX_FS = 6;
    AX_FW = 'Normal';
    LB_FS = 7;
    LB_FW = 'Normal';
    AX_LW = 0.5;
    PL_LW = 0.8;
    PL_MS = 4;
    AR_FS = 5;
elseif strcmp(type,'plos')==1
    AX_FN = 'Arial';
    LB_FN = 'Arial';
    AX_FS = 8;
    AX_FW = 'Normal';
    LB_FS = 8;
    LB_FW = 'Normal';
    AX_LW = 0.5;
    PL_LW = 1;
    PL_MS = 6;
    AR_FS = 5;    
    % the rest is deprecated
elseif strcmp(type,'visres')==1
    AX_FS = 7;
    AX_FW = 'Normal';
    LB_FS = 7;
    LB_FW = 'Bold';
    AX_LW = 0.5;
    PL_LW = 1;
    PL_MS = 6;
elseif strcmp(type,'jov')==1
    AX_FS = 10;
    AX_FW = 'Normal';
    LB_FS = 12;
    LB_FW = 'Normal';
    AX_LW = 0.5;
    PL_LW = 1;
    PL_MS = 6;
    AR_FS = 5;
elseif strcmp(type,'nature')==1
    AX_FS = 7;
    AX_FW = 'Normal';
    LB_FS = 7;
    %LB_FS = 9; %changed for S_R on 12.04.2010
    LB_FW = 'Normal';
    AX_LW = 0.5;
    PL_LW = 1;
    PL_MS = 6;
    AR_FS = 5;
else
    error('Unknown Document Type');
end

%% Axes
av = findall(fh,'Type','Legend');
for i=1:length(av)
    ah = av(i);  
    formatLegend(ah);
end
av = findall(fh,'Type','Axes');
for i=1:length(av)
    ah = av(i);
    if strcmp(get(ah,'Tag'),'legend')==1
        formatLegend(ah);
    else
        set(ah,'TickDir','out');
        set(ah,'FontSize',AX_FS);
        if isempty(getappdata(ah,'FontName'))
            set(ah,'FontName',AX_FN);
        end
        set(ah,'LineWidth',AX_LW);
        set(ah,'FontAngle',AX_FA);
        %if strcmp(type,'nature')
        if isempty(getappdata(ah,'MinorTickAxis'))
            if (strcmp(get(ah,'Tag'),'Colorbar')~=1) && (isempty(getappdata(fh,'PrepareFigure')))
                pbratio = get(ah,'PlotBoxAspectRatio');
                if max(pbratio)>1 % correct for non-square axis
                    tickLength = get(ah,'TickLength');
                    tickLength(1) = tickLength(1)/max(pbratio);
                    set(ah,'TickLength',tickLength);
                end
                tickLength = get(ah,'TickLength');
                tickLength(1) = tickLength(1)*2;
                set(ah,'TickLength',tickLength);
            end
            if isempty(getappdata(ah,'XMinorTick'))
                set(ah,'XMinorTick','off');
            end
            if isempty(getappdata(ah,'YMinorTick'))
                set(ah,'YMinorTick','off');
            end
        end
        if strcmp(get(ah,'Tag'),'Colorbar')~=1
            set(ah,'Box','off');
            %normalizeLogAxis(ah);
        end       
        if ~isempty(getappdata(ah,'normalizeDecimalPlaces'))
            normalizeDecimalPlaces(ah);
        end
        %         elseif isempty(getappdata(ah,'PostProcessing'))
        %             set(ah,'Box','on');
        %         end
        breakAxis(ah,AX_LW.*1.5);

        %% Labels
        lh = get(ah,'XLabel');
        set(lh,'FontSize',LB_FS);
        set(lh,'FontWeight',LB_FW);
        set(lh,'FontName',LB_FN);
        set(lh,'FontAngle',LB_FA);
        lh = get(ah,'YLabel');
        set(lh,'FontSize',LB_FS);
        set(lh,'FontWeight',LB_FW);
        set(lh,'FontName',LB_FN);
        set(lh,'FontAngle',LB_FA);
        %% Title
        th = get(ah,'Title');
        set(th,'FontSize',LB_FS);
        set(th,'FontWeight',LB_FW);
        set(th,'FontName',LB_FN);        
        set(th,'FontAngle',LB_FA);

        %% Lines
        lv = findall(ah,'Type','Line');
        lv = [lv; findall(ah,'Type','errorbar')];
        for j=1:length(lv)
            lh = lv(j);
            if isempty(getappdata(lh,'PostProcessing'))
                set(lh,'LineWidth',PL_LW);
                set(lh,'MarkerSize',PL_MS);
            elseif strcmp(getappdata(lh,'PostProcessing'),'PsychometricData')==1
                set(lh,'LineWidth',PL_LW);
                set(lh,'MarkerSize',PL_MS*0.75);
            elseif strcmp(getappdata(lh,'PostProcessing'),'PsychometricFunction')==1
                set(lh,'LineWidth',PL_LW*1.5);
                set(lh,'MarkerSize',PL_MS*0.5);
            elseif strcmp(getappdata(lh,'PostProcessing'),'AxisLineWidth')==1
                set(lh,'LineWidth',AX_LW);
            end
        end

        %% Text
        tv = findobj(ah,'Type','Text');
        for j=1:length(tv)
            th = tv(j);
            if isempty(getappdata(th,'PostProcessing'))
            if isempty(getappdata(th,'FontTemplate'))
                set(th,'FontSize',LB_FS);
                set(th,'FontWeight',LB_FW);
                set(th,'FontName',LB_FN);
                set(th,'FontAngle',LB_FA);
            elseif strcmp(getappdata(th,'FontTemplate'),'Axis') == 1
                set(th,'FontSize',AX_FS);
                set(th,'FontWeight',AX_FW);
                set(th,'FontName',AX_FN);
                set(th,'FontAngle',AX_FA);
            elseif strcmp(getappdata(th,'FontTemplate'),'AxisTwoThird') == 1
                set(th,'FontSize',AX_FS/3*2);
                set(th,'FontWeight',AX_FW);
                set(th,'FontName',AX_FN);
                set(th,'FontAngle',AX_FA);                
            elseif strcmp(getappdata(th,'FontTemplate'),'AxisHalf') == 1
                set(th,'FontSize',AX_FS*0.5);
                set(th,'FontWeight','Bold');
                set(th,'FontName',AX_FN);
                set(th,'FontAngle',AX_FA);
            end                   
            end
            %if strcmpi(type,'nature') && ~isempty(getappdata(th,'subindex'))
            if ~isempty(getappdata(th,'subindex'))
                set(th,'FontSize',LB_FS*1.5);
                if strcmpi(type,'plos')
                    set(th,'FontWeight','bold');
                end
            end
        end

        %% Annotations
        v = findobj(ah,'Type','hgtransform');
        for j=1:length(v)
            h = v(j);
            set(h,'LineWidth',PL_LW);
            set(h,'HeadLength',AR_FS);
            set(h,'HeadWidth',AR_FS);
            if strcmp(getappdata(h,'PostProcessing'),'AxisLineWidth')==1
                set(lh,'LineWidth',AX_LW);            
            end
        end
        v = findobj(ah,'Type','hggroup');
        for j=1:length(v)
            h = v(j);
            set(h,'LineWidth',PL_LW);          
            if strcmp(getappdata(h,'PostProcessing'),'AxisLineWidth')==1
                set(h,'LineWidth',AX_LW);            
            end
        end        

        %% Panel labels
        v = findobj(ah,'tag','subindex');
        for j=1:length(v)
            h = v(j);
            set(h,'FontSize',LB_FS);
            set(h,'FontWeight',LB_FW);
            set(h,'FontName',LB_FN);
            set(h,'FontAngle',LB_FA);
        end
    end
end
setappdata(fh,'PrepareFigure','on');

    function formatLegend(ah)
        set(ah,'Box','off');
        set(ah,'FontSize',AX_FS);
        set(ah,'FontName',AX_FN);
        set(ah,'FontAngle',AX_FA);

        if ~isempty(getappdata(ah,'LegendSize'))
            lc = get(ah,'Children');
            ln = length(lc);
            lm = lc(1:3:ln);
            ll = lc(2:3:ln);
            lt = lc(3:3:ln);
            for l=1:length(lt)              
                tpos = get(lt(l),'Position');
                xnew = tpos(1)*str2num(getappdata(ah,'LegendSize'));
                set(lm(l),'Xdata',xnew);
                
                lpos = get(ll(l),'Xdata');
                lpos = [xnew-(lpos(2)-xnew) lpos(2)];
                set(ll(l),'Xdata',lpos);
            end
            ydiff = (get(lm(2),'Ydata')-get(lm(1),'YData'));%           
            ydel = (ydiff.*(1-str2num(getappdata(ah,'LegendSize')))).*(length(lt)-1);            
            ydiff = ydiff.*str2num(getappdata(ah,'LegendSize'));
            for l=1:length(lt)
                if l==1
                    ypos = get(lm(1),'Ydata')+ydel;
                else
                    ypos = get(lm(l-1),'Ydata')+ydiff;
                end
                set(lm(l),'Ydata',ypos);
                set(ll(l),'Ydata',[ypos ypos]);
                tpos = get(lt(l),'Position');
                tpos(2) = ypos;
                set(lt(l),'Position',tpos);
            end
%             lpos = get(ah,'Position');
%             lpos(3)  = lpos(3).*0.2;%str2num(getappdata(ah,'LegendSize'));
%             lpos(4)  = lpos(4).*0.2;%str2num(getappdata(ah,'LegendSize'));
%             set(ah,'Position',lpos);
            for l=1:length(lm)
                %set(lm(l),'Xdata',get(lm(l),'Xdata')*1.25);
            end
        end
        % cheat for repositioning of legend
        if ~isempty(getappdata(ah,'LegendLocation'))
            legendLocation = getappdata(ah,'LegendLocation');
            set(ah,'Location',legendLocation);
        end                
    end

    function normalizeLogAxis(ah)
        if strcmpi(get(ah,'XScale'),'log') && strcmpi(get(ah,'XTickMode'),'auto')
            tickLabels = get(ah,'XTickLabel');
            tickLabels = 10.^str2num(tickLabels);
            set(ah,'XTickLabel',tickLabels);
        end
        if strcmpi(get(ah,'YScale'),'log') && strcmpi(get(ah,'YTickMode'),'auto')
            tickLabels = get(ah,'YTickLabel');
            tickLabels = 10.^str2num(tickLabels);
            set(ah,'YTickLabel',tickLabels);
        end
    end

    function normalizeDecimalPlaces(ah)
        for ai=1:2
            switch ai
                case 1
                    tick = get(ah,'YTick');
                case 2
                    tick = get(ah,'XTick');
            end
            for ti=1:length(tick)
                p(ti)=0;
                while roundn(tick(ti),p(ti))~=tick(ti)
                    p(ti)=p(ti)+1;
                end
            end
            pmax = max(p);
            for ti=1:length(tick)
                format = sprintf('%%.%df',pmax);
                if tick(ti)==0
                    format = sprintf('%%.%df',0);
                elseif tick(ti)<0
                    format = sprintf('–%%.%df',pmax);
                    tick(ti) = tick(ti).*-1;
                end
                ticklabel{ti} = sprintf(format,tick(ti));
            end
            switch ai
                case 1
                    set(ah,'YTickLabel',ticklabel);
                case 2
                    set(ah,'XTickLabel',ticklabel);
            end
        end
    end

    function breakAxis(ah, lw)
        DEBUG = 0;
        color = 'c';
        color = [0.99 0.99 0.99];
        xlim = get(ah,'XLim');
        ylim = get(ah,'YLim');
        xtick = get(ah,'XTick');
        ytick = get(ah,'YTick');
        shift = ~isempty(getappdata(gcf,'breakAxis'));
        lw_x = points2fu(lw*0.5,'h');
        lw_y = points2fu(lw*0.5,'v');

        % Draw rectangle for debugging
        if DEBUG
            if ~isempty(getappdata(ah,'breakAxis'))
                x = xlim;
                y = ylim;
                [x y] = dsxy2figxy(ah,x,y);
                if shift
                    x = x-lw_x;
                end
                v = [x(1) y(1) x(2)-x(1) y(2)-y(1)];
                h = annotation('rectangle',v);
                set(h,'Color','c','LineWidth',lw);
            end
        end
        if ~isempty(strfind(getappdata(ah,'breakAxis'),'xl')) && xtick(1)~=xlim(1)
            x = [xlim(1) xtick(1)];
            if strcmp(get(ah,'XAxisLocation'),'bottom')==1
                y = [ylim(1) ylim(1)];
            else
                y = [ylim(2) ylim(2)];
            end
            [x y] = dsxy2figxy(ah,x,y);
            if shift
                %x = x-lw_x;
                x(2) = x(2)-lw_x*2;
            else
                x(1) = x(1)+lw_x;
                x(2) = x(2)-lw_x;
            end
            h = annotation('line',x,y);
            set(h,'Color',color,'LineWidth',lw*2);
        end
        if ~isempty(strfind(getappdata(ah,'breakAxis'),'xr')) && xtick(end)~=xlim(2)
            x = [xtick(end) xlim(2)];
            if strcmp(get(ah,'XAxisLocation'),'bottom')==1
                y = [ylim(1) ylim(1)];
            else
                y = [ylim(2) ylim(2)];
            end
            [x y] = dsxy2figxy(ah,x,y);
            x(1) = x(1)+lw_x;
            h = annotation('line',x,y);
            set(h,'Color',color,'LineWidth',lw*2);
        end
        if ~isempty(strfind(getappdata(ah,'breakAxis'),'yb'))
            if strcmp(get(ah,'YAxisLocation'),'left')==1
                x = [xlim(1) xlim(1)];
            else
                x = [xlim(2) xlim(2)];
            end
            y = [ylim(1) ytick(1)];
            [x y] = dsxy2figxy(ah,x,y);
            if shift
                x = x-lw_x;
            end
            y = y-lw_y*1.5;
            %y(2) = y-lw_y*0.5;
            h = annotation('line',x,y);
            set(h,'Color',color,'LineWidth',lw*3);
        end
        if ~isempty(strfind(getappdata(ah,'breakAxis'),'yt'))
            if strcmp(get(ah,'YAxisLocation'),'left')==1
                x = [xlim(1) xlim(1)];
            else
                x = [xlim(2) xlim(2)];
            end
            y = [ytick(end) ylim(2)];
            [x y] = dsxy2figxy(ah,x,y);
            if shift
                x = x-lw_x;
            end
            %y = y+lw_y*1.5;
            %y(1) = y+lw_y*0.5;
            %y = y+lw_y*4;
            y(1) = y(1)+lw_y*2;
            y(2) = y(2)-lw_y*2;
            h = annotation('line',x,y);
            set(h,'Color',color,'LineWidth',lw*3);
        end
    end

    function fu = points2fu(points,dir)
        units = get(gcf,'Units');
        set(gcf,'Units','points');
        pos = get(gcf,'Position');
        if strcmp(dir,'h')
            fu = points./pos(3);
        elseif strcmp(dir,'v')
            fu = points./pos(4);
        end
        set(gcf,'Units',units)
    end
end