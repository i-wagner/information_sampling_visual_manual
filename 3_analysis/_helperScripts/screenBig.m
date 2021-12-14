function screen = screenBig()

    % Returns settings for screen in big lab
    % Output
    % screen: structure, with screen parameters (size, screen/subject
    %         distance) as well as pix2deg conversion factor

    %% Define screen resolution and screen center
    screen.x_pix    = 1920;
    screen.y_pix    = 1080;
    screen.x_center = screen.x_pix / 2;
    screen.y_center = screen.y_pix / 2;


    %% Define screen size and screen/subject distance
    screen.x_cm       = 90.7;
    screen.y_cm       = 51;
    screen.vp_dist_cm = 106;


    %% Define the factor to convert pixel to degree visual angle
    screen.xPIX2DEG = atan((screen.x_cm / screen.x_pix) / screen.vp_dist_cm) * (180 / pi);
    screen.yPIX2DEG = atan((screen.y_cm / screen.y_pix) / screen.vp_dist_cm) * (180 / pi);

end