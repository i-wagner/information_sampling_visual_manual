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