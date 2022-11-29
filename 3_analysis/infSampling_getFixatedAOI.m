function aoiVisit = infSampling_getFixatedAOI(endpoints_x, endpoints_y, stimLoc_x, stimLoc_y, aoi_radius, flag_bg, debug_plot)

    % Determines if and on which AOI a gaze shift landed
    % Input
    % endpoints_:    vector with gaze shift endpoints, for which we want to
    %                determine the AOI it landed on. Rows are individual
    %                gaze shifts
    % stimLoc_:      row-vector with coordinates of stimuli, for which we
    %                want to check if a gaze shift landed on them. Columns
    %                should be indivdiual stimuli, if multiple stimuli were
    %                shown in a trial. Not-shown stimuli should be NaN
    % aoi_radius:    radius of AOI around stimulus center; whenever the
    %                eyes land within the area around the stimulus, described
    %                by the radius, we count this as the stimulus being
    %                fixated
    % flag_bg:       flag to mark gaze shifts that landed outside any AOI,
    %                i.e., the background
    % debug_plot:    flag, toggels if plots of gaze shift endpoints and
    %                stimulus locations should be created
    % Output
    % aoiVisit:     column-vector with index of stimulus, on which a gaze
    %               shift landed. The index corresponds to the location of
    %               the stimulus coordinates in the coordinates-vector,
    %               provided as input

    %% Determine if participant fixated any AOI
    % Loop through all detected gaze shifts and check if they landed within
    % any AOI
    unitCircle = 0:0.01:2*pi;
    no_gs      = size(endpoints_x, 1);
    no_stim    = numel(stimLoc_x);
    aoiVisit   = NaN(no_gs, no_stim);
    for gs = 1:no_gs % Gaze shift

        for stim = 1:no_stim % Stimulus

            % Get location of current stimulus "stim"
            currStim_x  = stimLoc_x(stim);
            currStim_y  = stimLoc_y(stim);
            coord_aoi_x = cos(unitCircle) .* aoi_radius + currStim_x;
            coord_aoi_y = sin(unitCircle) .* aoi_radius + currStim_y;

            % Determine if gaze shift landed on any AOI, and if it did, on
            % which AOI; to do this, we take the mean gaze position after
            % the gaze shift, until the next gaze shift
            li_gsInAoi = inpolygon(endpoints_x(gs), endpoints_y(gs), coord_aoi_x, coord_aoi_y);
            if li_gsInAoi == 1 % Gaze shift landed on an AOI

                aoiVisit(gs, stim) = stim;

            end
            clear currStim_x currStim_y coord_aoi_x coord_aoi_y li_gsInAoi

        end
        clear stim

    end
    clear gs unitCircle no_gs no_stim

    % Sanitiy check: did any gaze shift land on more than one AOI?
    if any(sum(~isnan(aoiVisit), 2) > 1)

        keyboard

    else

        % Gaze shift that landed on an AOI will get the index of the
        % corresponding stimulus in the location vector as a flag
        aoiVisit = nansum(aoiVisit, 2);

        % Gaze shifts that landed on the background will get an obvious,
        % arbitrary flag. A gaze shift landed on the background if it has
        % both, an onset and an offset, and if it did not land on any of
        % the defined AOIs
        li_visitBg = aoiVisit == 0 & ~isnan(endpoints_x) & ~isnan(endpoints_y);

        aoiVisit(li_visitBg)    = flag_bg;
        aoiVisit(aoiVisit == 0) = NaN;     % Gaze shift without offset
        clear li_visitBg

    end


    %% Plot position all stimuli and saccade endpoints
    if debug_plot(1)

        % Compose vector with area, covered by screen, as well as locations
        % of stimuli in a trial
        x_screen_aoi = [-24.5129, -24.5129, 24.5129,  24.5129,  -24.5129]; % Screen area
        y_screen_aoi = [-13.7834,  13.7834, 13.7834, -13.78341, -13.7834];
        FIXTOLAAREA  = 2;
        if debug_plot(2) == 2

            y_screen_aoi = [(-13.7834+9.5), 27.5668, 27.5668, (-13.7834+9.5), (-13.7834+9.5)];
            FIXTOLAAREA  = 1.50;

        end
        x_screen_stim = [];
        y_screen_stim = [];

        unitCircle = 0:0.01:2*pi;
        no_stim    = numel(stimLoc_x);
        for s = 1:no_stim

            currStim_x   = stimLoc_x(s);
            currStim_y   = stimLoc_y(s);
            coord_aoi_x  = cos(unitCircle) .* aoi_radius + currStim_x;
            coord_aoi_y  = sin(unitCircle) .* aoi_radius + currStim_y;
            coord_stim_x = cos(unitCircle) .* (1.2512/2) + currStim_x;
            coord_stim_y = sin(unitCircle) .* (1.2512/2) + currStim_y;
            clear currStim_x currStim_y

            x_screen_aoi  = [x_screen_aoi NaN coord_aoi_x];
            y_screen_aoi  = [y_screen_aoi NaN coord_aoi_y];
            x_screen_stim = [x_screen_stim coord_stim_x NaN];
            y_screen_stim = [y_screen_stim coord_stim_y NaN];
            clear coord_aoi_x coord_aoi_y coord_stim_x coord_stim_y

        end
        clear s no_stim

        % Check which gaze shift offset ended somewhere on the screen,
        % outside any AOI.
        % CAUTION: "inpolygons" ASSUMES THAT HOLES IN THE BACKGROUND ARE
        % ARRANGED CCW (which is the case here by default). IF WE WOULD JUST
        % PUT IN THE COORDINATES OF THE AOIs, WITHOUT FIRST REARRANGING THEM
        % PROPERLY, "inpolygons" WOULD FAIL TO DETERMINE THAT ANY OFFSET FELL
        % WITHIN ANY AOI, EVEN IF an OFFSET WAS LOCATED WITHIN AN AOI
        % (BECAUSE, WITHOUT REARRANGING THEM CW, THE AOIs ARE TREATED AS HOLES).
        % IF WE ONLY PUT IN THE COORDINATES OF THE AOIs, WE HAVE TO FIRST ARRANGE
        % THEM CW WITH "poly2cw"
        [li_in, idx_in] = inpolygons(endpoints_x, endpoints_y, x_screen_aoi, y_screen_aoi);
        idx_in          = cell2mat(idx_in);

        % Create plot
%         close all
        clf
        [f, v] = poly2fv(x_screen_aoi, y_screen_aoi);                           % Plot screen area and AOIs
        patch('Faces',     f, ...
              'Vertices',  v, ...
              'FaceColor', [.9 .9 .9], ...
              'EdgeColor', 'none');
        hold on
        plot(x_screen_stim, y_screen_stim)                                      % Plot area, occupied by stimulus
        hold off
        daspect([1 1 1]);

        hold on
        plot(0, 0, '+', 'MarkerSize', 15)                               % Plot fixation cross
        plot(cos(unitCircle).*FIXTOLAAREA+0, ...
             sin(unitCircle).*FIXTOLAAREA+0)                            % Plot fixation tolerance area
        plot(endpoints_x(li_in),       endpoints_y(li_in), 'r.', ...    % Plot endpoints inside/outside any AOI
             endpoints_x(~li_in),      endpoints_y(~li_in), 'b.');
        plot(endpoints_x(idx_in == 1), endpoints_y(idx_in == 1), 'go'); % Plot corona around endpoints outside any AOI
        text(endpoints_x, endpoints_y, num2str((1:numel(endpoints_x))'));
        hold off
        waitforbuttonpress

    end

end