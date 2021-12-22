function epar = exp_self_start(epar)

    %% Draw text
    DrawFormattedText(epar.window,sprintf('Press any key to start experiment.'), ...
                      'center', epar.fixLoc_px(2), epar.black, 80, [], [], 1.75);
    Screen('Flip', epar.window);
    WaitSecs(0.2);


    %% Proceed after button press
    while 1

        [epar.keyIsDown, ~, ~, ~] = KbCheck([]);
        if epar.keyIsDown

            break

        end

    end

end