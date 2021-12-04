function exp_mon_exit(epar)

    %% Restore the initial Gamma
    if epar.GAMMA

        Screen('LoadNormalizedGammaTable', epar.window, epar.oldGamma);

    end


    %% Close all windows
    Screen('Close', epar.window);
    Screen('CloseAll')


    %% Show the cursor again
    ShowCursor;

end