%% Load screen settings
screen.distance.subject.CM = 106; % Distance between screen and subject

screen.size.x.CM = 90.7;
screen.size.y.CM = 51;

screen.size.x.PX = 1920;
screen.size.y.PX = 1080;
screen.center.x.PX = screen.size.x.PX / 2;
screen.center.y.PX = screen.size.y.PX / 2;

%% Define conversion factor
screen.pix2deg.X = atand(screen.size.x.CM / screen.size.x.PX / ...
                         screen.distance.subject.CM);
screen.pix2deg.Y = atand(screen.size.y.CM / screen.size.y.PX / ...
                         screen.distance.subject.CM);

%% Define reference point, relative to which we calculate dva values
% Dva values are NOT calculated relative to screen center (which is 
% convention), because the fixation cross was not placed at the screen 
% center, but the lower part of the screen
screen.referencePoint.x.PX = exper.fixation.location.x.PX;
screen.referencePoint.y.PX = exper.fixation.location.y.PX;

%% Define screen bounds
% Screen bounds correspond to the most extreme gaze coordinates that we can
% measure, and which are still located on the area covered by the screen
screen.bounds.dva.X = ...
    (screen.size.x.PX - exper.fixation.location.x.PX) * screen.pix2deg.X;
screen.bounds.dva.Y = ...
    [exper.fixation.location.y.PX * screen.pix2deg.Y, ...
     (screen.size.y.PX - exper.fixation.location.y.PX) * screen.pix2deg.Y * -1];