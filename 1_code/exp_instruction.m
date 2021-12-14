function exp_instruction(epar)

    %% Set instruction text
    if ismember(epar.subject, epar.sub_blueE)

        str_inst = 'Blaue hat ein groessere Luecke als Rot!';

    else

        str_inst = 'Rot hat ein groessere Luecke als Blau!';

    end


    %% Generate some example stimuli
    horStim = randperm(size(epar.targHor_idx, 2), 1);
    verStim = randperm(size(epar.targVert_idx, 2), 1);

    % Get indices of to-be-shown targets
    epar.stim.targ_idxE = epar.stim.targ_e(epar.targHor_idx(1, horStim), epar.diff2);
    epar.stim.targ_idxD = epar.stim.targ_d(epar.targVert_idx(1, verStim), epar.diff3);

    % Define screen coordinates
    x1 = epar.x_center - 50;
    x2 = epar.x_center + 50;
    x3 = epar.x_center - 100;
    x4 = epar.x_center + 100;
    y  = epar.y_center + 50;

    % Define rect
    epar.texture.rect1 = CenterRectOnPoint([0 0 epar.pic_size epar.pic_size], x1, y);
    epar.texture.rect2 = CenterRectOnPoint([0 0 epar.pic_size epar.pic_size], x2, y);
    epar.texture.rect3 = CenterRectOnPoint([0 0 epar.pic_size epar.pic_size], x3, y);
    epar.texture.rect4 = CenterRectOnPoint([0 0 epar.pic_size epar.pic_size], x4, y);


    %% Show instruction screen
    Screen('FillRect', epar.window, epar.gray);
    DrawFormattedText(epar.window, str_inst, 'center', ...
                      'center', epar.black, 100, [], [], 1.75);                                         % Instruction text
    Screen('DrawTexture', epar.window, epar.stim.targ_idxE, [], epar.texture.rect1, [], 0);             % Easy target
    Screen('DrawTexture', epar.window, epar.stim.targ_idxD, [], epar.texture.rect2, [], 0);             % Difficult target
    Screen('DrawTexture', epar.window, epar.stim.dist_e_br(epar.diff2), [], epar.texture.rect3, [], 0); % Easy distractor
    Screen('DrawTexture', epar.window, epar.stim.dist_d_tl(epar.diff3), [], epar.texture.rect4, [], 0); % Difficult distractor
    Screen('Flip', epar.window);


    %% Wait for a buttonpress, before flipping to the next screen
    while 1

        [~, keyCode] = KbWait([], 2);
        if keyCode(KbName('Return'))

            break

        end

    end
    Screen('FillRect', epar.window, epar.gray);
    Screen('Flip', epar.window);

end