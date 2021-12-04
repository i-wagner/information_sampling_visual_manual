function [ex, ey] = exp_el_eye_pos(el)

    eye_used = Eyelink('EyeAvailable'); % Get eye that's tracked
    if eye_used == el.BINOCULAR         % If both eyes are tracked

        eye_used = el.LEFT_EYE; % Use left eye

    end


    %% Get the sample in the form of an event structure
    ex = NaN;
    ey = NaN;
    if Eyelink('NewFloatSampleAvailable') > 0

        evt = Eyelink('NewestFloatSample');
        if eye_used ~= -1 % Do we know which eye to use yet?

            % If we do, get current gaze position from sample
            x = evt.gx(eye_used + 1); % +1 as we're accessing MATLAB array
            y = evt.gy(eye_used + 1);

            % Do we have valid data and is the pupil visible?
            if x ~= el.MISSING_DATA & y ~= el.MISSING_DATA & evt.pa(eye_used + 1) > 0

                ex = x;
                ey = y;

            end

        end

    end

end