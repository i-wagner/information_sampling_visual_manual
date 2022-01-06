%% Eye link settings
epar.EL          = 1; % 1: Eye link active; 0: No eye link
epar.GAMMA       = 1; % 1: Gamma correction; 0: No correction
epar.GAMMA_TABLE = 'C:\SRC\lut\01065_propixx_pc11485_hscon'; % Big lab


%% Data storage settings (save path, running path)
epar.EXPNAME      = 'informationSampling2022';
epar.save_path    = ('./data/');
epar.running_path = ('.');


%% Screen settings
epar.SCREEN_X_PIX = 1920;
epar.SCREEN_Y_PIX = 1080;
epar.x_center     = epar.SCREEN_X_PIX / 2;
epar.y_center     = epar.SCREEN_Y_PIX / 2;

epar.screen_x_cm = 90.7;
epar.screen_y_cm = 51;
epar.vp_dist_cm  = 106;

epar.XPIX2DEG = atan((epar.screen_x_cm / epar.SCREEN_X_PIX) / epar.vp_dist_cm) * (180/pi);
epar.YPIX2DEG = atan((epar.screen_y_cm / epar.SCREEN_Y_PIX) / epar.vp_dist_cm) * (180/pi);

epar.SAMP_FREQ    = 1000;
epar.MONITOR_FREQ = 120;


%% General Settings
epar.CALIB_TRIALS = 0;
epar.CALIB_X      = 12;                                                         % Max. x position for calibration dot
epar.CALIB_Y      = 12;                                                         % Max. y position for calibration dot
epar.fixLoc       = [0 9.5];                                                    % Onscreen location of fixation cross (relative to screen center)
epar.fixLoc_px    = round([(epar.fixLoc(1) / epar.XPIX2DEG) + epar.x_center ... 
                           (epar.fixLoc(2) / epar.YPIX2DEG) + epar.y_center]);  % Pixel location of fixation cross
epar.fixsize      = round([0.15 0.60] ./ epar.XPIX2DEG);                        % Size of the fixation-cross
epar.fix_min      = 0.50;                                                       % Minimum fixation interval duration
epar.fix_max      = 1;                                                          % Maximum fixation interval duration
epar.fix_tol      = 1.5 / epar.XPIX2DEG;                                        % Tolerance threshold for fixation


%% Unify key names for Mac OS and Windows
KbName('UnifyKeyNames');