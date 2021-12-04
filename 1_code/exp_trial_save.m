function exp_trial_save(epar, tn)

    fid = fopen(epar.log_file, 'a');
    fprintf(fid, '%d\t %d\t %d\t %d\t %d\t    %d\t %d\t %d\t %d\t %d\t    %d\t %d\t %d\t %d\t %.4f\t    %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t    %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t    %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t    %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t    %d\t %.4f\t %d\t %d\t %d\t    %.4f\n', ...
        epar.expNo, ...
        epar.subject, ...
        epar.block, ...
        tn, ...
        epar.trials.targ(tn), ... 5; No. of targets shown in a trial
        epar.diff(tn), ... Target (easy/hard) shown in a trial (only Experiment 1 and 2)
        epar.trials.stairSteps(tn, 1), ... Difficulty level (easy target)
        epar.trials.stairSteps(tn, 2), ... Difficulty level (hard target)
        epar.stim.gap(tn, 1), ... Gap location (Easy)
        epar.stim.gap(tn, 2), ... 10; Gap location (Hard)
        epar.stim.gapResp(tn), ... Reported gap location
        epar.trials.dist_e(tn), ... No. of easy distractor in trial
        epar.trials.dist_d(tn), ... No. of hard distractor in trial
        epar.trials.dist_num(tn), ... Overall no. of distractors in trial
        epar.x_pick(tn, :), ... 15:24; Drawn screen positions on x-axis
        epar.y_pick(tn, :), ... 25:34; Drawn screen positions on y-axis
        epar.timer_cum, ... 35; Cumulative timer
        epar.perf.hit(tn), ...
        epar.score, ... Score
        epar.fix_error, ... Not fixated during stimulus onset
        epar.time(1), ... Fixation onset
        epar.time(2), ... 40; Onset of stimuli
        epar.time(3)); % Response onset
    fclose(fid);

end