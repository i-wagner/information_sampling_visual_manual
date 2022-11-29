function color = plotColors

    % Loads some default colors, so I do not have to define them manually
    % every single time
    % Output
    % color: structure, with some default colors

    %% Define colors
    % Red
    color.r1 = [1   0.5 0.5]; 
    color.r2 = [0.8 0.2 0.2];
    color.r3 = [0.6 0.1 0.1];

    % Blue
    color.b1 = [0.1 0.6 1];
    color.b2 = [0.2 0.2 0.5];
    color.b3 = [0.1 0.1 0.5];

    % Green
    color.g1 = [0.5 0.8 0.5];
    color.g2 = [0.3 0.6 0.3];
    color.g3 = [0.1 0.6 0.1];

    % Grey
    color.c1    = [0.5 0.5 0.5];
    color.c2    = [0.7 0.7 0.7];

    % Black/white
    color.black = [0 0 0];
    color.white = [1 1 1]; 

end