function epar = exp_self_start(epar)

    %% Draw text
    DrawFormattedText(epar.window,sprintf('Press Enter/Space to start experiment.'), ...
                      'center', epar.fixLoc_px(2), epar.black, 80, [], [], 1.75);
    Screen('Flip', epar.window);
    WaitSecs(0.2);


    %% Proceed after button press
    while 1

        [~, keyCode] = KbWait([], 2);
        if keyCode(KbName('Return')) | keyCode(KbName('Space'))

            Screen('FillRect', epar.window, epar.gray);
            Screen('Flip', epar.window);
            break

        end

    end

end