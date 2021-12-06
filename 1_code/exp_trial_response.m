function epar = exp_trial_response(epar, el, tn)

    %% Get AOIs in trial (returns AOI area in deg)
    [xCircS, yCircS] = defineAOIs(epar.x_pick(tn, :), epar.y_pick(tn, :), 5, epar.pic_size, epar.XPIX2DEG);

    % Reshape AOI coordinates into a row vector and prepare for "inpolygons"
    xCircS_row = reshape(xCircS.', 1, []);
    yCircS_row = reshape(yCircS.', 1, []);
    [xCircS_row, yCircS_row] = poly2cw_mr(xCircS_row, yCircS_row);


    %% Record a participant's response, time of the response and track dwell time
    dispStim_rectOn       = zeros(1, numel(epar.stim.txt_disp_mask));                     % Helper variable; track for which stimulus rectangle was turned on
    dispStim_maskOn       = zeros(1, numel(epar.stim.txt_disp_mask));                     % Helper variable; track for which stimulus mask is shown definitely
    dispStim_maxDwellTime = zeros(1, numel(epar.stim.txt_disp_mask)) + epar.maxDwellTime; % Helper variable; track how much viewing time is left for stimuli
    txt_disp_replace      = epar.stim.txt_disp_mask;                                      % Helper variable; dynamically repalce stimuli with rectangle/mask
%     bb                    = 1;
    time_current          = NaN(5000, 1);
    time_idx              = 1;
    idx_lastFixatedAOI    = NaN;

    response_start_time = GetSecs;
    while 1

        % Get current gaze position
        [ex, ey] = exp_el_eye_pos(el);

        % Check what was fixated during trial: show rectangles if gaze is in
        % AOI and some dwel time is left, otherwise show mask
        time_current(time_idx) = GetSecs;
        if ~isnan(ex)

            % Check if gaze is within any AOI
            ex = (ex - epar.x_center) .* epar.XPIX2DEG; % Transform gaze coordinates to deg
            ey = (ey - epar.y_center) .* epar.YPIX2DEG;

            [in, idx] = inpolygons(ex, ey, xCircS_row, yCircS_row);

            idx = cell2mat(idx);
            if iscell(in)

                in = cell2mat(in);

            end

            % If gaze is located within one of the AOIs, check if it landed
            % in the AOI for the first time or if it dwelled within the AOI
            % for a while already
            if in == 1

                if isnan(idx_lastFixatedAOI) % Gaze arrived in AOI for first time

                    % Save arrival time and idx of AOI, in which gaze landed
                    time_arrivedInAOI  = GetSecs; % Arrival time
                    idx_lastFixatedAOI = idx;     % Index

                elseif idx == idx_lastFixatedAOI % Gaze stayed in AOI

                    % Calculate how long gaze is within AOI
                    time_dwellTime = time_current(time_idx) - time_arrivedInAOI;

                    % Display rectangle if some dwell time for stimulus is
                    % left, show mask otherwise
                    fixedStimDwellTime = dispStim_maxDwellTime(idx_lastFixatedAOI);
                    if fixedStimDwellTime > 0 && time_dwellTime < fixedStimDwellTime % Dwell time left

                        % Update remaining maximum dwell time of last fixated stimulus
                        dispStim_maxDwellTime(idx_lastFixatedAOI) = ...
                            dispStim_maxDwellTime(idx_lastFixatedAOI) - ...
                                (time_current(time_idx) - time_arrivedInAOI);

                        % Check if we are showing mask or rectangle
                        if ~dispStim_rectOn(idx_lastFixatedAOI)

                            txt_disp_replace(idx_lastFixatedAOI) = epar.stim.txt_disp(idx_lastFixatedAOI);

                            exp_target_draw(epar.window, epar.x_center, epar.y_center, ...
                                            epar.fixsize(2), epar.fixsize(1), epar.fixcol, epar.gray);
                            Screen('DrawTextures', epar.window, ...
                                   txt_disp_replace(1:length(txt_disp_replace)), [], ...
                                   epar.tex_rect(:, 1:length(txt_disp_replace)), [], 0);
                            Screen('Flip', epar.window);
%                             if epar.EL
% 
%                                 Eyelink('Message', 'MAX_DWELL_EXC');
% 
%                             end
                            
                            dispStim_rectOn(idx_lastFixatedAOI) = 1;

                        end

                    elseif fixedStimDwellTime <= 0 && ~dispStim_maskOn(idx_lastFixatedAOI) % No dwell time left

                        txt_disp_replace(idx_lastFixatedAOI) = epar.stim.txt_disp_mask(idx_lastFixatedAOI);

                        exp_target_draw(epar.window, epar.x_center, epar.y_center, ...
                                        epar.fixsize(2), epar.fixsize(1), epar.fixcol, epar.gray);
                        Screen('DrawTextures', epar.window, ...
                               txt_disp_replace(1:length(txt_disp_replace)), [], ...
                               epar.tex_rect(:, 1:length(txt_disp_replace)), [], 0);
                        Screen('Flip', epar.window);
%                         if epar.EL
% 
%                             Eyelink('Message', 'MAX_DWELL_EXC');
% 
%                         end

                        dispStim_maskOn(idx_lastFixatedAOI) = 1;

                    end

                elseif idx ~= idx_lastFixatedAOI % Gaze went on to new AOI

                    dispStim_maxDwellTime(idx_lastFixatedAOI) = ...
                        dispStim_maxDwellTime(idx_lastFixatedAOI) - ...
                            (time_current(time_idx-1) - time_arrivedInAOI);

                    % Save arrival time and idx of the stimulus, on which
                    % gaze landed
                    time_arrivedInAOI  = GetSecs;
                    idx_lastFixatedAOI = idx;

                end

            elseif in == 0 % If gaze landed outside of an AOI (i.e., the background)

                % Update dwell time for last fixated stimulus
                if ~isnan(idx_lastFixatedAOI)

                    dispStim_maxDwellTime(idx_lastFixatedAOI) = ...
                        dispStim_maxDwellTime(idx_lastFixatedAOI) - ...
                            (time_current(time_idx-1) - time_arrivedInAOI);

                end

                % Reset time of arrival in AOI as well as idx of fixated stimulus
                idx_lastFixatedAOI = NaN;
                time_arrivedInAOI  = NaN;

            end

        end
        time_idx = time_idx + 1;

        % Record participants response
        % In Experiment 2, only continue if the pressed key corresponds to
        % the stimulus orientation; i.e., if the stimulus is orientad vertically,
        % either the left or right key has to be preessed in order to proceed
        [~, ~, keyCode, ~] = KbCheck([]);
        if keyCode(KbName('DownArrow'))

            response_end_time = GetSecs;
            epar.stim.gapResp(tn) = 1;
            if epar.expNo == 2

                if epar.stim.gap(tn, 1) == 1 || epar.stim.gap(tn, 1) == 2

                    break

                end

            else

                break

            end

        elseif keyCode(KbName('UpArrow'))

            response_end_time= GetSecs;
            epar.stim.gapResp(tn) = 2;
            if epar.expNo == 2

                if epar.stim.gap(tn, 1) == 1 || epar.stim.gap(tn, 1) == 2

                    break

                end

            else

                break

            end

        elseif keyCode(KbName('LeftArrow'))

            response_end_time = GetSecs;
            epar.stim.gapResp(tn) = 3;
            if epar.expNo == 2

                if epar.stim.gap(tn, 1) == 3 || epar.stim.gap(tn, 1) == 4

                    break

                end

            else

                break

            end

        elseif keyCode(KbName('RightArrow'))

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

        % Visualize gaze and save frames
%         pixbound_x= 2/epar.XPIX2DEG;
%         pixbound_y= 2/epar.YPIX2DEG;
%         exp_target_draw(epar.window, epar.x_center, epar.y_center, ...
%                         epar.fixsize(2), epar.fixsize(1), epar.fixcol, epar.gray);
%         Screen('DrawTextures', epar.window, ...
%                txt_disp_replace(1:length(txt_disp_replace)), [], ...
%                epar.tex_rect(:, 1:length(txt_disp_replace)), [], 0);
%         [x, y] = exp_el_eye_pos (el);
%         Screen ('FrameRect', epar.window, [0 0 0], [x-pixbound_x y-pixbound_y x+pixbound_x y+pixbound_y], []) 
%         Screen('Flip', epar.window);
% 
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
    if length(dispStim_maxDwellTime) < 10

        epar.stim.maxDwellTime(tn, :) = [dispStim_maxDwellTime NaN(1, 10-length(dispStim_maxDwellTime))];

    else

        epar.stim.maxDwellTime(tn, :) = dispStim_maxDwellTime;

    end
    

    %% Set script priority back to initial level
    Priority(1);

end