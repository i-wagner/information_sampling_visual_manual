function exp_example_draw(w, x, y, r, fg, angle)

    if nargin < 6 || isempty(angle)

        angle = 45;

    end

    angle    = (angle + 90) / 180 * pi;
    [xd, yd] = pol2cart([angle angle], [r/2 -r/2]);

    Screen('DrawLine', w, fg, x+xd(1), y+yd(1), x+xd(2), y+yd(2), 2);

end