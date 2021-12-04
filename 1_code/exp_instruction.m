function exp_instruction(epar)

    %% Read instruction text
    Screen('FillRect', epar.window, epar.gray);
    fid = fopen(sprintf('exp_instruction_%d.txt', epar.expNo), 'rb');
    instruction = fread(fid, [1, inf], 'char');
    instruction = char(instruction);
    fclose(fid);


    %% Display the instruction text and the example stimuli
    % Draw instruction text
    DrawFormattedText(epar.window, sprintf(instruction), 'center', ...
                      'center', epar.black, 100, [], [], 1.75);

    % Randomly select if gap is left/right, up/down
    horStim = randperm(size(epar.targHor_idx, 2), 1);
    verStim = randperm(size(epar.targVert_idx, 2), 1);

    % Get indices of to-be-shown targets
    epar.stim.targ_idxE = epar.stim.targ_e(epar.targHor_idx(1, horStim), ...
                                           epar.diff2);
    epar.stim.targ_idxD = epar.stim.targ_d(epar.targVert_idx(1, verStim), ...
                                           epar.diff3);

    % Define screen coordinates
    x1 = epar.x_center - 150;
    x2 = epar.x_center + 150;
    x3 = epar.x_center - 300;
    x4 = epar.x_center + 300;
    y  = epar.y_center + 450;

    % Define rect
    epar.texture.rect1 = CenterRectOnPoint([0 0 epar.pic_size epar.pic_size], x1, y);
    epar.texture.rect2 = CenterRectOnPoint([0 0 epar.pic_size epar.pic_size], x2, y);
    epar.texture.rect3 = CenterRectOnPoint([0 0 epar.pic_size epar.pic_size], x3, y);
    epar.texture.rect4 = CenterRectOnPoint([0 0 epar.pic_size epar.pic_size], x4, y);

    % Draw stimuli
    Screen('DrawTexture', epar.window, epar.stim.targ_idxE, [], epar.texture.rect1, [], 0); % Easy target
    Screen('DrawTexture', epar.window, epar.stim.targ_idxD, [], epar.texture.rect2, [], 0); % Hard target
    Screen('DrawTexture', epar.window, epar.stim.dist_e_br(epar.diff2), [], epar.texture.rect3, [], 0);
    Screen('DrawTexture', epar.window, epar.stim.dist_d_tl(epar.diff3), [], epar.texture.rect4, [], 0);
    Screen('Flip', epar.window);

    % Wait for a buttonpress, before flipping to the next screen
    while 1

        [~, keyCode] = KbWait([], 2);
        if keyCode(KbName('Space'))

            break

        end

    end
    Screen('FillRect', epar.window, epar.gray);
    Screen('Flip', epar.window);

end