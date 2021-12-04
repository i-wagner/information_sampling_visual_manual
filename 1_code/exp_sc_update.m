function epar = exp_sc_update(epar, tn)

    %% Check if trial was hit/miss
    % Hit, when the reported gap position matches the actual gap position
    if epar.expNo == 2

        if epar.stim.gap(tn, 1) == epar.stim.gapResp(tn)

            epar.perf.hit(tn) = 1;

        else

            epar.perf.hit(tn) = 0;

        end

    else

        if epar.stim.gap(tn, 1) == epar.stim.gapResp(tn) || ...
           epar.stim.gap(tn, 2) == epar.stim.gapResp(tn)

            epar.perf.hit(tn) = 1;

        else

            epar.perf.hit(tn) = 0;

        end

    end


    %% Add/subtract point
    if epar.perf.hit(tn) == 1 && epar.fix_error == 0

        epar.score = epar.score + epar.rewardD;

    else

        epar.score = epar.score - epar.rewardD;

    end


    %% Show feedback
    if epar.fix_error == 1 % Didn't fixate at trial start

        DrawFormattedText(epar.window, sprintf('Nicht fixiert!'), 'center', ...
                          'center', epar.black, [],[],[], 1.75);

    else % Everything's allright

        DrawFormattedText(epar.window, sprintf('%.2f | %.2f', round(epar.perf.hit(tn)/50 .* 2 - 0.02, 2), ...
                          round(epar.score, 2)), 'center', 'center', epar.black, ...
                          [],[],[], 1.75);

        % Draw remaining Experiment duration
        DrawFormattedText(epar.window, sprintf('%s', duration([0, 0, epar.duration]) - ...
                          duration([0, 0, epar.timer_cum])), 'center', 575, ...
                          epar.black, [],[],[], 1.75);

    end
    time1 = Screen('Flip', epar.window);
    Screen('FillRect', epar.window, epar.gray);
    Screen('Flip', epar.window, time1 + epar.feedback_dur);

end