function exp_el_start(el, t, x, y)

    Eyelink('command', 'set_idle_mode');
    WaitSecs(0.05);
    fprintf('\nTRIALID: %d; ',t);
    success = EyelinkDoDriftCorrection(el, x, y, 1, 1);
    fprintf('Drift: %d; ', success);
    Eyelink('command', 'set_idle_mode');
    WaitSecs(0.05);
    error = Eyelink('StartRecording');
    fprintf('Recording: %d; ',error);

    if error

        WaitSecs(0.5);
        error = Eyelink('StartRecording');
        fprintf('Recording: %d; ',error);

    end

    Eyelink('Message', ['TrialID' sprintf('%d',t)]);
    WaitSecs(0.1);

end