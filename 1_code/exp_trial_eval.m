function epar = exp_trial_eval(epar)

    %% Define placeholder variables
    epar.fix_error = 0;


    %% Evaluate eye movements
    if ~isempty(dir(epar.eye_name))

        % Get events in current trial
        data    = dlmread(epar.eye_name);
        display = find(bitand(data(:, 5), 4));

        % Define the recorded eye
        if mean(abs(data(:, 2))) > 32000

            col.x = 6;
            col.y = 7;

            col.event = 9;

        else

            col.x = 2;
            col.y = 3;

            col.event = 5;

        end

        % Check if the gaze was at the screen center, when the target(s) appeared
        fix_time          = data(display(4)-20:display(4)+80, col.x:col.y);
        fix_time(:, 1)    = fix_time(:, 1) - epar.fixLoc_px(1); 
        fix_time(:, 2)    = fix_time(:, 2) - epar.fixLoc_px(2);
        fix_deviate_frame = sum(sum(abs(fix_time) > epar.fix_tol));

        if fix_deviate_frame > 0

            epar.fix_error = 1;

        end

    end

end