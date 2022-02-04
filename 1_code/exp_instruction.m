function exp_instruction(epar)

    %% Set instruction text
    if epar.stim.diffFlag_blueEasy

        str_inst = 'Blau hat ein groessere Luecke als Rot!';

    else

        str_inst = 'Rot hat ein groessere Luecke als Blau!';

    end
    str_inst = [str_inst '\nNutzen Sie Ihre NICHT-DOMINANTE Hand zum antworten!'];


    %% Generate some example stimuli
    idx_horStim = randperm(numel(epar.targHor_idx), 1);
    idx_verStim = randperm(numel(epar.targVert_idx), 1);

    % Get indices of to-be-shown targets
    targ_easy = epar.stim.targ_e(epar.targHor_idx(idx_horStim), epar.targDiff_easy);
    targ_hard = epar.stim.targ_d(epar.targVert_idx(idx_verStim), epar.targDiff_hard);

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
    DrawFormattedText(epar.window, str_inst, ...
                      'center', 'center', epar.black, 100, [], [], 1.75);                                       % Instruction text
    Screen('DrawTexture', epar.window, targ_easy, [], epar.texture.rect1, [], 0);                               % Easy target
    Screen('DrawTexture', epar.window, targ_hard, [], epar.texture.rect2, [], 0);                               % Difficult target
    Screen('DrawTexture', epar.window, epar.stim.dist_e_bl(epar.targDiff_easy), [], epar.texture.rect3, [], 0); % Easy distractor
    Screen('DrawTexture', epar.window, epar.stim.dist_d_tl(epar.targDiff_hard), [], epar.texture.rect4, [], 0); % Difficult distractor
    Screen('Flip', epar.window);


    %% Wait for a buttonpress, before flipping to the next screen
    while 1

        [~, keyCode] = KbWait([], 2);
        if keyCode(KbName('Return')) | keyCode(KbName('Space'))

            Screen('FillRect', epar.window, epar.gray);
            Screen('Flip', epar.window);
            break

        end

    end

end