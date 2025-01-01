function prepareFigure(fig_h, opt)

    % Saves figure in a publish-ready format
    % Input
    % fig_h: handle of the figure, one wants to plot (for example,
    %        "fig_h = figure(1)"). Can also be the handle to a figure,
    %        containing multiple subplots
    % opt:   options for behavior and appearance of figure

    %% Check input
    % Just use the defaults if no figure options are provided 
    if ~exist('opt', 'var'); opt = []; end


    %% Set defaults
    if ~isfield(opt, 'save'),        opt.save        = 0;           end % Save figure to file or not
    if ~isfield(opt, 'position'),    opt.position    = [5 5];       end % Figure position on screen
    if ~isfield(opt, 'size'),        opt.size        = [15 15];     end % Figure size
    if ~isfield(opt, 'imgname'),     opt.imgname     = 'name_me';   end % File name of saved figure
    if ~isfield(opt, 'format'),      opt.format      = '-dpng';     end % File format of saved figure
    if ~isfield(opt, 'res'),         opt.res         = '-r300';     end % Resolution (dpi)
    if ~isfield(opt, 'fontName'),    opt.fontName    = 'Helvetica'; end % Font name
    % if ~isfield(opt, 'fontWeight'),  opt.fontWeight  = 'normal';    end % Font weight
    if ~isfield(opt, 'fontSize'),    opt.fontSize    = 15;          end % Font size
    if ~isfield(opt, 'doLines'),     opt.doLines     = 0;           end % 
    if ~isfield(opt, 'LineWidth'),   opt.LineWidth   = 2;           end % Linewidth of plot axes
    if ~isfield(opt, 'tickDir'),     opt.tickDir     = 'out';       end % Linewidth of plot ticks
    if ~isfield(opt, 'axLayer'),     opt.axLayer     = 'top';       end % Layer of axes
    if ~isfield(opt, 'axLineWidth'), opt.axLineWidth = 2;           end % Width of axis lines


    %% Adjust renderer for eps files
    % If we save an eps file we have to switch the renderer to "painters"
    % (default: OpenGL), otherwise some figures might be saved as pixel
    % instead of vector files (happend to me before, for example, when I
    % tried to save a histogram as an eps, without changing the renderer)
    if strcmp(opt.format, '-deps') || strcmp(opt.format, '-depsc')

        set(fig_h, 'Renderer', 'painters');

    end


    %% Get all axes handles from input figure
    % If the input figure is composed of multiple subplots, this command
    % will get the handles of all individual subplots, allowing us to
    % change their properties (font, etc.) all at once
    axes_h = findobj(fig_h, 'Type', 'Axes');


    %% Get all legend handles from input figure
    leg_h = findobj(fig_h, 'Type', 'Legend');


    %% Adjust figure properties
    set(findall([axes_h; leg_h], '-property', 'FontName'),   'FontName',   opt.fontName)   % Fontname
    if isfield(opt, 'fontWeight')
        set(findall([axes_h; leg_h], '-property', 'FontWeight'), 'FontWeight', opt.fontWeight) % Fontweight
    end
    set(findall([axes_h; leg_h], '-property', 'FontSize'),   'FontSize',   opt.fontSize)   % Fontsize
    if opt.doLines == 1

        % This one takes all lines (illustrative lines, axes lines, lines
        % between datapoints) and applies property to them
        set(findall(axes_h, '-property', 'LineWidth'), 'LineWidth', opt.LineWidth) % Width of lines

    else

        % Change properties of axis from all subplots in figure 
        no_sp = numel(axes_h);
        for sp = 1:no_sp % Subplot

            % Put axis lines on top layer
            axes_h(sp).Layer = 'bottom';

            % Change width of axis lines
            % The "axle" properties only become accessible once the 
            % graphics are fully rendered; use "drawnow" to ensure that
            % this is the case before we try to access the property
            drawnow;
            axes_h(sp).XRuler.Axle.LineWidth = opt.axLineWidth;
            axes_h(sp).YRuler.Axle.LineWidth = opt.axLineWidth;

            % Change width of tick lines
            % We can have more than one y-axis on aplot panel (i.e., left
            % and right y-axis). We account for this by checking the numbr
            % of y-axis in a panel
            axes_h(sp).XAxis.MajorTickChild.LineWidth = opt.axLineWidth;

            nAxY = numel(axes_h(sp).YAxis);
            for y = 1:nAxY
                axes_h(sp).YAxis(y).MajorTickChild.LineWidth = opt.axLineWidth;
            end

        end

    end
    set(findall(axes_h, '-property', 'Layer'),   'Layer',   opt.axLayer); % Location of plot axes, relative to other elements in plot
    set(findall(axes_h, '-property', 'TickDir'), 'TickDir', opt.tickDir); % Axis ticks are outside (i.e., outside the data-area)


    %% Save figure
    % Settings
    set(fig_h, ...
        'Units', 'centimeters')               % Change units to cm (not pixel)
    set(fig_h, ...
        'Position', [opt.position opt.size])  % Change position and size of the figure
    pause(1); % Wait a bit to give the program enough time to resize figure

    % Save figure to drive
    if opt.save

        % print(fig_h, ...
        %       opt.imgname, ...
        %       opt.format, ...
        %       opt.res)
        % In order to make this function still compatible with the print()
        % function, avoid changing format of save parameters in the init
        % block, but change change them here instead
        exportgraphics(fig_h, ...
                       strcat(opt.imgname, '.', opt.format(3:end)), ...
                       'Resolution', str2double(regexprep(opt.res, '[^0-9\s]','')));

    end

end