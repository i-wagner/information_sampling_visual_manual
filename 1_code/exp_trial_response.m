function epar = exp_trial_response(epar, el, tn)

    %% DEBUG; show AOIs
%     Screen('DrawLine', epar.window, epar.black, ...
%            0,                 epar.fixLoc_px(2) - round(((min(epar.y) + ((max(epar.y) - min(epar.y)) / 2)) / epar.YPIX2DEG)), ...
%            epar.SCREEN_X_PIX, epar.fixLoc_px(2) - round(((min(epar.y) + ((max(epar.y) - min(epar.y)) / 2)) / epar.YPIX2DEG)))
%     Screen('DrawLine', epar.window, epar.black, ...
%            epar.x_center, 0, ...
%            epar.x_center, epar.SCREEN_Y_PIX)
%     Screen('DrawLine', epar.window, epar.black, ...
%            epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%            epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
%     Screen('DrawLine', epar.window, epar.black, ...
%            epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%            epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
%     Screen('DrawLine', epar.window, epar.black, ...
%            epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%            epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG)
%     Screen('DrawLine', epar.window, epar.black, ...
%            epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG, ...
%            epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
%     x_test = round(epar.fixLoc_px(1) + (epar.x_pick(tn, :) ./ epar.XPIX2DEG));
%     y_test = round(epar.fixLoc_px(2) - (epar.y_pick(tn, :) ./ epar.YPIX2DEG));
% 
%     centeredRect = NaN(4, numel(x_test));
%     for r = 1:numel(x_test) % Stimulus rect
% 
%         baseRect           = [0 0 epar.aoiSize/epar.XPIX2DEG epar.aoiSize/epar.XPIX2DEG];
%         centeredRect(:, r) = CenterRectOnPoint(baseRect, x_test(r), y_test(r))';
% 
%     end
%     exp_target_draw(epar.window, epar.fixLoc_px(1), epar.fixLoc_px(2), ...
%                     epar.fixsize(2), epar.fixsize(1), epar.fixcol, epar.gray);
%     Screen('DrawTextures', epar.window, epar.stim.txt_disp_mask, [], epar.tex_rect);
%     Screen('FrameOval', epar.window, epar.black, centeredRect);
%     Screen('Flip', epar.window);

    % Debuging variables
    % Used to check when stimuli are turned on/off and to make screenshots
%     test  = [];
%     test2 = [];
%     test3 = [];
%     bb    = 1;


    %% Get AOIs in trial (returns AOI area in deg)
    [xCircS, yCircS] = defineAOIs(epar.x_pick(tn, :), epar.y_pick(tn, :), ...
                                  epar.aoiSize, epar.pic_size, epar.XPIX2DEG);

    % Reshape AOI coordinates into a row vector and prepare for "inpolygons"
    xCircS_row = reshape(xCircS.', 1, []);
    yCircS_row = reshape(yCircS.', 1, []);

    [xCircS_row, yCircS_row] = poly2cw_mr(xCircS_row, yCircS_row);


    %% Record a participant's response, time of the response and track dwell time
    stimToShow             = epar.stim.txt_disp_mask;                                      % Dynamic variable that tracks which stimulus is shown as mask/rectangle
    remainingDwellTime     = zeros(1, numel(epar.stim.txt_disp_mask)) + epar.maxDwellTime; % Helper variable; tracks how much viewing time is left for stimuli
    fixatedAoi_dwellExceed = zeros(1, numel(epar.stim.txt_disp_mask));                     % Helper variable; tracks for which stimuli allowed dwell time was exceeded
    fixatedAoi_currently   = NaN;                                                          % Helper variable; tracks which AOI was fixated in last loop iteration
    fixatedAoi_arrivalTime = NaN(numel(epar.stim.txt_disp_mask), 1);                       % Helper variable; tracks time when gaze entered a new AOI
    fixatedAoi_departTime  = NaN(numel(epar.stim.txt_disp_mask), 1);                       % Helper variable; tracks when gaze left a ficated AOI
% ShowCursor('Arrow');
    response_start_time = GetSecs;
    while 1

        % Get which AOI was fixated during last check
        fixatedAoi_last = fixatedAoi_currently;

        % Gaze contigent display
        [ex, ey, flag_missingSample] = exp_el_eye_pos(el);
% [ex, ey] = GetMouse(epar.window);
        ex                           = (ex - epar.fixLoc_px(1)) .* epar.XPIX2DEG;
        ey                           = (epar.fixLoc_px(2) - ey) .* epar.YPIX2DEG;
        if ~isnan(ex)

            % Get which AOI is fixated during current check
            [in, fixatedAoi_currently] = inpolygons(ex, ey, xCircS_row, yCircS_row);
            fixatedAoi_currently       = cell2mat(fixatedAoi_currently);
            li_outsideAoi              = fixatedAoi_currently == 0;

            fixatedAoi_currently(li_outsideAoi) = NaN; % Set AOI ID NaN if gaze is outside any defined AOI (i.e., on screen background)

            % Get time when AOI was entered/left for cases in which AOI
            % changed between two checks
            if in && fixatedAoi_currently ~= fixatedAoi_last

                fixatedAoi_arrivalTime(fixatedAoi_currently) = GetSecs;

            elseif ~in && fixatedAoi_currently ~= fixatedAoi_last && ~isnan(fixatedAoi_last)

                fixatedAoi_departTime(fixatedAoi_last) = GetSecs;

            end

            % Update remaining dwell time
            % Only if fixated AOI was changed between two checks and the
            % last fixated AOI was not the background
            if ~isnan(fixatedAoi_last) && fixatedAoi_last ~= fixatedAoi_currently

                remainingDwellTime(fixatedAoi_last) = ...
                        remainingDwellTime(fixatedAoi_last) - (GetSecs - fixatedAoi_arrivalTime(fixatedAoi_last));

            end

            % Check if we have to show a rectangle
            % Rectangles are shown when a new AOI is fixated and there is
            % some dwell time left for this AOI
            if ~isnan(fixatedAoi_currently) && ...
               fixatedAoi_currently ~= fixatedAoi_last && ...
               remainingDwellTime(fixatedAoi_currently) > 0

                stimToShow(fixatedAoi_currently) = epar.stim.txt_disp(fixatedAoi_currently);

                exp_target_draw(epar.window, epar.fixLoc_px(1), epar.fixLoc_px(2), ...
                                epar.fixsize(2), epar.fixsize(1), epar.fixcol, epar.gray);
% Screen('FrameOval', epar.window, epar.black, centeredRect);
% Screen('DrawLine', epar.window, epar.black, ...
%        0,                 epar.fixLoc_px(2) - round(((min(epar.y) + ((max(epar.y) - min(epar.y)) / 2)) / epar.YPIX2DEG)), ...
%        epar.SCREEN_X_PIX, epar.fixLoc_px(2) - round(((min(epar.y) + ((max(epar.y) - min(epar.y)) / 2)) / epar.YPIX2DEG)))
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.x_center, 0, ...
%        epar.x_center, epar.SCREEN_Y_PIX)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
                Screen('DrawTextures', epar.window, ...
                       stimToShow(1:length(stimToShow)), [], ...
                       epar.tex_rect(:, 1:length(stimToShow)), [], 0);
                Screen('Flip', epar.window);
% test = [test; GetSecs];

            end

            % Check if maximum dwell time of currently fixated AOI is
            % exceeded and rectangle has to be turned off;
            if fixatedAoi_currently == fixatedAoi_last && ...
               all(~isnan([fixatedAoi_currently fixatedAoi_last])) && ...
               (GetSecs - fixatedAoi_arrivalTime(fixatedAoi_last)) > epar.maxDwellTime && ...
               ~fixatedAoi_dwellExceed(fixatedAoi_last)

                stimToShow(fixatedAoi_currently)        = epar.stim.txt_disp_mask(fixatedAoi_currently);
                fixatedAoi_dwellExceed(fixatedAoi_last) = 1;

                exp_target_draw(epar.window, epar.fixLoc_px(1), epar.fixLoc_px(2), ...
                                epar.fixsize(2), epar.fixsize(1), epar.fixcol, epar.gray);
% Screen('FrameOval', epar.window, epar.black, centeredRect);
% Screen('DrawLine', epar.window, epar.black, ...
%        0,                 epar.fixLoc_px(2) - round(((min(epar.y) + ((max(epar.y) - min(epar.y)) / 2)) / epar.YPIX2DEG)), ...
%        epar.SCREEN_X_PIX, epar.fixLoc_px(2) - round(((min(epar.y) + ((max(epar.y) - min(epar.y)) / 2)) / epar.YPIX2DEG)))
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.x_center, 0, ...
%        epar.x_center, epar.SCREEN_Y_PIX)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
                Screen('DrawTextures', epar.window, ...
                       stimToShow(1:length(stimToShow)), [], ...
                       epar.tex_rect(:, 1:length(stimToShow)), [], 0);
                Screen('Flip', epar.window);
% test2 = [test2; GetSecs];

            end

            % Check if rectangle in previously fixated AOI has to be turned
            % off; it is not turned off immediately after gaze leaves an
            % AOI, but after some delay, to avoid flickering
            currTime = GetSecs;
            li_turnOff = (currTime - fixatedAoi_departTime) > epar.stimTurnOffOffset;
            if any(li_turnOff)

% test3 = [test3; GetSecs]; sca; keyboard
                stimToShow(li_turnOff)            = epar.stim.txt_disp_mask(li_turnOff);
                fixatedAoi_departTime(li_turnOff) = NaN;

                exp_target_draw(epar.window, epar.fixLoc_px(1), epar.fixLoc_px(2), ...
                                epar.fixsize(2), epar.fixsize(1), epar.fixcol, epar.gray);
% Screen('FrameOval', epar.window, epar.black, centeredRect);
% Screen('DrawLine', epar.window, epar.black, ...
%        0,                 epar.fixLoc_px(2) - round(((min(epar.y) + ((max(epar.y) - min(epar.y)) / 2)) / epar.YPIX2DEG)), ...
%        epar.SCREEN_X_PIX, epar.fixLoc_px(2) - round(((min(epar.y) + ((max(epar.y) - min(epar.y)) / 2)) / epar.YPIX2DEG)))
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.x_center, 0, ...
%        epar.x_center, epar.SCREEN_Y_PIX)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
                Screen('DrawTextures', epar.window, ...
                       stimToShow(1:length(stimToShow)), [], ...
                       epar.tex_rect(:, 1:length(stimToShow)), [], 0);
                Screen('Flip', epar.window);

            end

        elseif isnan(ex) & flag_missingSample

            % Update remaining dwell time
            if ~isnan(fixatedAoi_last)

                remainingDwellTime(fixatedAoi_last) = ...
                        remainingDwellTime(fixatedAoi_last) - (GetSecs - fixatedAoi_arrivalTime(fixatedAoi_last));

            end

            % Show masks when gaze signal is lost
            stimToShow = epar.stim.txt_disp_mask;

            exp_target_draw(epar.window, epar.fixLoc_px(1), epar.fixLoc_px(2), ...
                            epar.fixsize(2), epar.fixsize(1), epar.fixcol, epar.gray);
% Screen('FrameOval', epar.window, epar.black, centeredRect);
% Screen('DrawLine', epar.window, epar.black, ...
%        0,                 epar.fixLoc_px(2) - round(((min(epar.y) + ((max(epar.y) - min(epar.y)) / 2)) / epar.YPIX2DEG)), ...
%        epar.SCREEN_X_PIX, epar.fixLoc_px(2) - round(((min(epar.y) + ((max(epar.y) - min(epar.y)) / 2)) / epar.YPIX2DEG)))
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.x_center, 0, ...
%        epar.x_center, epar.SCREEN_Y_PIX)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - min(epar.y) / epar.YPIX2DEG)
% Screen('DrawLine', epar.window, epar.black, ...
%        epar.fixLoc_px(1) - min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG, ...
%        epar.fixLoc_px(1) + min(epar.x) / epar.XPIX2DEG, epar.fixLoc_px(2) - max(epar.y) / epar.YPIX2DEG)
            Screen('DrawTextures', epar.window, ...
                   stimToShow(1:length(stimToShow)), [], ...
                   epar.tex_rect(:, 1:length(stimToShow)), [], 0);
            Screen('Flip', epar.window);

        end

% Kill switch
% [~, ~, keyCode, ~] = KbCheck([]);
% if keyCode(KbName('q')); Eyelink('StopRecording'); sca; keyboard; end

        % Record participants response
        % In Experiment 2, only continue if the pressed key corresponds to
        % the stimulus orientation; i.e., if the stimulus is orientad vertically,
        % either the left or right key has to be preessed in order to proceed
        [~, ~, keyCode, ~] = KbCheck([]);
        if keyCode(KbName('2'))

            response_end_time = GetSecs;
            epar.stim.gapResp(tn) = 1;
            if epar.expNo == 2

                if epar.stim.gap(tn, 1) == 1 || epar.stim.gap(tn, 1) == 2

                    break

                end

            else

                break

            end

        elseif keyCode(KbName('8'))

            response_end_time = GetSecs;
            epar.stim.gapResp(tn) = 2;
            if epar.expNo == 2

                if epar.stim.gap(tn, 1) == 1 || epar.stim.gap(tn, 1) == 2

                    break

                end

            else

                break

            end

        elseif keyCode(KbName('4'))

            response_end_time = GetSecs;
            epar.stim.gapResp(tn) = 3;
            if epar.expNo == 2

                if epar.stim.gap(tn, 1) == 3 || epar.stim.gap(tn, 1) == 4

                    break

                end

            else

                break

            end

        elseif keyCode(KbName('6'))

            response_end_time = GetSecs;
            epar.stim.gapResp(tn) = 4;
            if epar.expNo == 2

                if epar.stim.gap(tn, 1) == 3 || epar.stim.gap(tn, 1) == 4

                    break

                end

            else

                break

            end

        end

        % Visualize gaze position
%         pixbound_x= 0.5/epar.XPIX2DEG;
%         pixbound_y= 0.5/epar.YPIX2DEG;
%         exp_target_draw(epar.window, epar.fixLoc_px(1), epar.fixLoc_px(2), ...
%                         epar.fixsize(2), epar.fixsize(1), epar.fixcol, epar.gray);
%         Screen('DrawTextures', epar.window, ...
%                stimToShow(1:length(stimToShow)), [], ...
%                epar.tex_rect(:, 1:length(stimToShow)), [], 0);
%         Screen('FrameOval', epar.window, epar.black, centeredRect);
%         [x, y] = exp_el_eye_pos (el);
%         Screen ('FrameRect', epar.window, [0 0 0], [x-pixbound_x y-pixbound_y x+pixbound_x y+pixbound_y], []) 
%         Screen('Flip', epar.window);

         % Make screenshots
%         imageArray = Screen('GetImage', epar.window);
%         if epar.expNo == 2
% 
%             imwrite(imageArray, strcat('stimArray_exp2_', num2str(bb), '.jpg'))
% 
%         elseif epar.expNo == 3
% 
%             imwrite(imageArray, strcat('stimArray_exp3_', num2str(bb), '.jpg'))
% 
%         end
%         bb=bb+1;

    end
    epar.time(3) = Screen('Flip', epar.window);
    if epar.EL

        Eyelink('Message', 'STIM_RESP/OFF');

    end


    %% Calculate how long it took to respond and save remaining dwell time per stimulus
    epar.response_time(tn) = response_end_time - response_start_time;
    epar.stim.maxDwellTime(tn, :) = [remainingDwellTime NaN(1, 10-numel(remainingDwellTime))];
    

    %% Set script priority back to initial level
    Priority(1);

end