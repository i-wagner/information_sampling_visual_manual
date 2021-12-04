function el = exp_el_init(epar)

    %% Provide Eyelink with details about the graphics environment and perform some initializations
    % The information is returned in a structure that also contains useful
    % defaults and control codes (e.g. tracker state bit and Eyelink key values)
    if EyelinkInit(0) ~= 1

        return

    end

    el                        = EyelinkInitDefaults(epar.window);
    el.backgroundcolour       = epar.gray;
    el.foregroundcolour       = epar.red;
    el.calibrationtargetsize  = round(0.6 ./ epar.XPIX2DEG);
    el.calibrationtargetwidth = round(0.15 ./ epar.XPIX2DEG);
    el.angle                  = 45;
    el.targetbeep             = 0;
    el.feedbackbeep           = 0;
    EyelinkUpdateDefaults(el);


    %% Define calibration
    x = epar.x_center;
    y = epar.y_center;
    x_off = round(epar.CALIB_X / epar.XPIX2DEG);
    y_off = round(epar.CALIB_Y / epar.YPIX2DEG);
    calib = sprintf('%d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d %d, %d', ...
                    x, y, ...
                    x, y-y_off, ...
                    x, y+y_off, ...
                    x-x_off, y, ...
                    x+x_off, y, ...
                    x-x_off, y-y_off, ...
                    x+x_off, y-y_off, ...
                    x-x_off, y+y_off, ...
                    x+x_off, y+y_off);
    Eyelink('command', 'calibration_type = HV9');
    Eyelink('command', 'generate_default_targets = NO');
    Eyelink('command', sprintf('calibration_targets = %s', calib));
    Eyelink('command', sprintf('validation_targets = %s', calib));
    Eyelink('command', 'button_function 1 ''accept_target_fixation''');


    %% Define data to get from eye link
    Eyelink('command', 'file_sample_data=LEFT, RIGHT, GAZE, AREA');
    edfFile = sprintf('e%dv%db%d.edf', epar.expNo, epar.subject, epar.block);
    Eyelink('OpenFile', edfFile);
    Eyelink('WaitForModeReady', 1000);

end