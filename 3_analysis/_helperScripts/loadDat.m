function [datFile, data_loss] = loadDat(t, screen_x, screen_y)

    % Load .dat file from trial "t", restructures file based on measured
    % eye, checks for not-flagged blinks/data-loss and check if data-chunks
    % are missing
    % Input
    % t:         trial from which we want to load the .dat file
    % screen_x:  horizontal screen size (pixel)
    % screen_y:  vertical screen size(pixel)
    % Output
    % datFile:   .dat file from trial "t", containing timestamps, gaze trace
    %            and bits for recorded eye
    % data_loss: flag, marking if data-loss occured during trial. If there
    %            is no data-loss, variable is empty, otherwise it outputs
    %            the number of the current trial "t"

    %% Load .dat file
    name_trial = sprintf('trial%d.dat', t);
    datFile    = load(name_trial);


    %% Restructure the trial file, based on which eye was measured
    if all(datFile(:, 2:3) == -32768)                    % Right eye recorded   

        datFile(:, 2:3) = datFile(:, 6:7);
        datFile(:, 4)   = datFile(:, 5) + datFile(:, 9);
        datFile(:, 5:9) = [];

    else

        datFile(:, 4)   = datFile(:, 5) + datFile(:, 9); % Left eye recorded
        datFile(:, 5:9) = [];

    end


    %% Check for not-detected blinks/data-loss
    % We check if there is any datasample with coordiantes outside the
    % measurable screen area and if, for this sample, the bit for a blink/a
    % saccade is not turned on. If this is the case, we turn the bit on so
    % we can, later, detect the blink/data-loss
%     idx_missedBlink = find(datFile(:, 2) == -3270 & ...
%                            bitget(datFile(:, 4), 2) == 0);
    idx_missedBlink = find((datFile(:, 2) > screen_x & bitget(datFile(:, 4), 2) == 0 & bitget(datFile(:, 4), 1) == 0) | ...
                           (datFile(:, 2) < 0        & bitget(datFile(:, 4), 2) == 0 & bitget(datFile(:, 4), 1) == 0) | ...
                           (datFile(:, 3) > screen_y & bitget(datFile(:, 4), 2) == 0 & bitget(datFile(:, 4), 1) == 0) | ... 
                           (datFile(:, 3) < 0        & bitget(datFile(:, 4), 2) == 0 & bitget(datFile(:, 4), 1) == 0));
    if ~isempty(idx_missedBlink)

        datFile(idx_missedBlink, 4) = bitset(datFile(idx_missedBlink, 4), 2);

    end


    %% Check for data-loss in trial
    % Sometimes chunks of samples are not played back into the .dat file,
    % resulting in data-loss. We can easily identy those cases by checking
    % for discontinuities in the timestamps
    data_loss = [];
    if any(diff(datFile(:, 1)) ~= 1)

        data_loss = t;

    end

end