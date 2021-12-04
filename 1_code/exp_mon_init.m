function epar = exp_mon_init(epar)

    %% Screen setup
    % Toggle hot-spot correction and video scanning backlight
    if ~isempty(strfind(epar.GAMMA_TABLE, 'sbloff'))

        PsychDataPixx('Open');
        PsychDataPixx('DisableVideoScanningBacklight');
        PsychDataPixx('Close');

    elseif ~isempty(strfind(epar.GAMMA_TABLE, 'sblon'))

        PsychDataPixx('Open');
        PsychDataPixx('EnableVideoScanningBacklight');
        PsychDataPixx('Close');

    elseif ~isempty(strfind(epar.GAMMA_TABLE, 'hscon'))

        Datapixx('Open');
        Datapixx('EnableHotspotCorrection');
        Datapixx('RegWrRd')
        Datapixx('Close')

    elseif ~isempty(strfind(epar.GAMMA_TABLE, 'hscoff'))

        Datapixx('Open');
        Datapixx('DisableHotspotCorrection');
        Datapixx('RegWrRd')
        Datapixx('Close')

    end

    % Open window
    screenNumber = max(Screen('Screens'));
    PsychImaging('PrepareConfiguration');
    PsychImaging('AddTask', 'General', 'FloatingPoint16Bit');
    [epar.window, epar.screenRect] = PsychImaging('OpenWindow', 0, 128);

    % Set colors
    epar.gray  = GrayIndex(screenNumber, 0.5);
    epar.black = BlackIndex(screenNumber);
    epar.white = WhiteIndex(screenNumber);
    epar.red   = [255 0 0];
    epar.blue  = [0 0 255];

    % Set font
    Screen('TextFont', epar.window, 'Arial');
    Screen('TextSize', epar.window, 12);


    %% Load gamma correction
    if epar.GAMMA

        initmon(epar.GAMMA_TABLE);
        newGamma(:, 1) = dlmread([epar.GAMMA_TABLE '.r']);
        newGamma(:, 2) = dlmread([epar.GAMMA_TABLE '.g']);
        newGamma(:, 3) = dlmread([epar.GAMMA_TABLE '.b']);
        epar.newGamma  = newGamma ./ 255;
        epar.oldGamma  = Screen('LoadNormalizedGammaTable', epar.window, ...
                                epar.newGamma);

    else

        epar.newGamma = NaN;
        epar.oldGamma = NaN;

    end


    %% Hide mouse cursor
    HideCursor;


    %% Load stimuli, sort them in arrays and convert them to textures
    % Read-in the image files with the stimuli
    cd '_stim/'
    dirStim = dir('*.png');

    % Loop through folder content and process image files
    epar.stim.cueR = [];
    epar.stim.cueB = [];
    for n = 1:length(dirStim)

        img = imread(dirStim(n).name);
        img = preprocessStimuli(img, epar.stimCont); % Preprocess stimulus

        % Create texture
        epar.stim.txt(n) = Screen('MakeTexture', epar.window, img);

        % Sort stimuli as targets and distractors
        % Targets are defined by a "t" at the start of their filename
        if strncmpi(dirStim(n).name, 't', 1)

            epar.stim.targ(n) = epar.stim.txt(n);

            % Sort targets by difficulty
            if ~isempty(strfind(dirStim(n).name, 'blue')) % Blue targets

                if ismember(epar.subject, epar.sub_blueE) % Blue is easy for half of participants

                    epar.stim.targ_e(n) = epar.stim.txt(n);
                    if ~isempty(strfind(dirStim(n).name, '-s-')) % Seperate targets by gap position
                                                                 % Gap position is defined by the
                                                                 % abbrivation of a cardinal point in its filename

                        epar.stim.targ_e_b(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-w-'))

                        epar.stim.targ_e_l(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-e-'))

                        epar.stim.targ_e_r(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-n-'))

                        epar.stim.targ_e_t(n) = epar.stim.txt(n);

                    end

                else % Blue is hard for other half of participants

                    epar.stim.targ_d(n) = epar.stim.txt(n);
                    if ~isempty(strfind(dirStim(n).name, '-s-'))

                        epar.stim.targ_d_b(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-w-'))

                        epar.stim.targ_d_l(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-e-'))

                        epar.stim.targ_d_r(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-n-'))

                        epar.stim.targ_d_t(n) = epar.stim.txt(n);

                    end

                end

            else % Red targets

                if ismember(epar.subject, epar.sub_blueE) % Red is hard for half of participants

                    epar.stim.targ_d(n) = epar.stim.txt(n);
                    if ~isempty(strfind(dirStim(n).name, '-s-'))

                        epar.stim.targ_d_b(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-w-'))

                        epar.stim.targ_d_l(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-e-'))

                        epar.stim.targ_d_r(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-n-'))

                        epar.stim.targ_d_t(n) = epar.stim.txt(n);

                    end

                else % Red is easy for other half of participants

                    epar.stim.targ_e(n) = epar.stim.txt(n);
                    if ~isempty(strfind(dirStim(n).name, '-s-'))

                        epar.stim.targ_e_b(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-w-'))

                        epar.stim.targ_e_l(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-e-'))

                        epar.stim.targ_e_r(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-n-'))

                        epar.stim.targ_e_t(n) = epar.stim.txt(n);

                    end

                end

            end

        else % If a stimulus is not a target, it's a distractor

            epar.stim.dist(n) = epar.stim.txt(n);
            if ~isempty(strfind(dirStim(n).name, 'blue'))

                if ismember(epar.subject, epar.sub_blueE)

                    epar.stim.dist_e(n) = epar.stim.txt(n);
                    if ~isempty(strfind(dirStim(n).name, '-sw-'))

                        epar.stim.dist_e_bl(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-se-'))

                        epar.stim.dist_e_br(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-nw-'))

                        epar.stim.dist_e_tl(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-ne-'))

                        epar.stim.dist_e_tr(n) = epar.stim.txt(n);

                    end

                else

                    epar.stim.dist_d(n) = epar.stim.txt(n);
                    if ~isempty(strfind(dirStim(n).name, '-sw-'))

                        epar.stim.dist_d_bl(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-se-'))

                        epar.stim.dist_d_br(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-nw-'))

                        epar.stim.dist_d_tl(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-ne-'))

                        epar.stim.dist_d_tr(n) = epar.stim.txt(n);

                    end

                end

            else

                if ismember(epar.subject, epar.sub_blueE)

                    epar.stim.dist_d(n) = epar.stim.txt(n);
                    if ~isempty(strfind(dirStim(n).name, '-sw-'))

                        epar.stim.dist_d_bl(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-se-'))

                        epar.stim.dist_d_br(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-nw-'))

                        epar.stim.dist_d_tl(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-ne-'))

                        epar.stim.dist_d_tr(n) = epar.stim.txt(n);

                    end

                else

                    epar.stim.dist_e(n) = epar.stim.txt(n);
                    if ~isempty(strfind(dirStim(n).name, '-sw-'))

                        epar.stim.dist_e_bl(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-se-'))

                        epar.stim.dist_e_br(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-nw-'))

                        epar.stim.dist_e_tl(n) = epar.stim.txt(n);

                    elseif ~isempty(strfind(dirStim(n).name, '-ne-'))

                        epar.stim.dist_e_tr(n) = epar.stim.txt(n);

                    end

                end

            end

        end

        % Make empty stimuli (only circle, no rectangular element)
        % To create them, we just take the first red/blue stimulus in the
        % loop and remove the rectangular element from its center
        if isempty(epar.stim.cueB) && ~isempty(strfind(dirStim(n).name, 'blue'))

            img            = preprocessStimuli(img, 0);
            img_cueB       = img;
            epar.stim.cueB = Screen('MakeTexture', epar.window, img);

        elseif isempty(epar.stim.cueR) && ~isempty(strfind(dirStim(n).name, 'red'))

            img            = preprocessStimuli(img, 0);
            img_cueR       = img;
            epar.stim.cueR = Screen('MakeTexture', epar.window, img);

        end

        % Create composite image of target/distractor orientations
        if n == length(dirStim)

            % Images, we will use for composite
            imgNames = {'distractor_blue-ne-level01.png'; 'distractor_blue-nw-level01.png'; ...
                        'target_blue-e-level01.png';      'target_blue-n-level01.png'};
            
            % Create composite image
            % We do this in two steps: first, we combine all orientations,
            % second, we recude the contrast of the composite images.
            % Through this, we do not get ugly summation effects at
            % intersections of different orientations
            [~, comp] = createTargDisComposite(imgNames, img_cueB, img_cueR, 1, epar.gray);
            comp{1}   = preprocessStimuli(comp{1}, epar.stimCont);
            comp{2}   = preprocessStimuli(comp{2}, epar.stimCont);

            epar.stim.compB = Screen('MakeTexture', epar.window, comp{1});
            epar.stim.compR = Screen('MakeTexture', epar.window, comp{2});

        end

    end


    %% Create variables with red/blue stimuli
    epar.stim.cue  = [epar.stim.cueR epar.stim.cueB];
    epar.stim.comp = [epar.stim.compR epar.stim.compB];


    %% Create variables with easy/hard targets/distractors
    % Remove empty entries from the texture variable. Those were created in
    % the loop, because I am an idiot and indexed instead of just
    % concatenating
    % Targets
    epar.stim.txt      = epar.stim.txt(epar.stim.txt ~= 0);
    epar.stim.targ     = epar.stim.targ(epar.stim.targ  ~= 0);
    epar.stim.targ_e   = epar.stim.targ_e(epar.stim.targ_e ~= 0);
    epar.stim.targ_e_b = epar.stim.targ_e_b(epar.stim.targ_e_b ~= 0);
    epar.stim.targ_e_l = epar.stim.targ_e_l(epar.stim.targ_e_l ~= 0);
    epar.stim.targ_e_r = epar.stim.targ_e_r(epar.stim.targ_e_r ~= 0);
    epar.stim.targ_e_t = epar.stim.targ_e_t(epar.stim.targ_e_t ~= 0);
    epar.stim.targ_d   = epar.stim.targ_d(epar.stim.targ_d ~= 0);
    epar.stim.targ_d_b = epar.stim.targ_d_b(epar.stim.targ_d_b ~= 0);
    epar.stim.targ_d_l = epar.stim.targ_d_l(epar.stim.targ_d_l ~= 0);
    epar.stim.targ_d_r = epar.stim.targ_d_r(epar.stim.targ_d_r ~= 0);
    epar.stim.targ_d_t = epar.stim.targ_d_t(epar.stim.targ_d_t ~= 0);

    % Distractors
    epar.stim.dist      = epar.stim.dist(epar.stim.dist ~= 0);
    epar.stim.dist_e    = epar.stim.dist_e(epar.stim.dist_e ~= 0);
    epar.stim.dist_e_bl = epar.stim.dist_e_bl(epar.stim.dist_e_bl ~= 0);
    epar.stim.dist_e_br = epar.stim.dist_e_br(epar.stim.dist_e_br ~= 0);
    epar.stim.dist_e_tl = epar.stim.dist_e_tl(epar.stim.dist_e_tl ~= 0);
    epar.stim.dist_e_tr = epar.stim.dist_e_tr(epar.stim.dist_e_tr ~= 0);
    epar.stim.dist_d    = epar.stim.dist_d(epar.stim.dist_d ~= 0);
    epar.stim.dist_d_bl = epar.stim.dist_d_bl(epar.stim.dist_d_bl ~= 0);
    epar.stim.dist_d_br = epar.stim.dist_d_br(epar.stim.dist_d_br ~= 0);
    epar.stim.dist_d_tl = epar.stim.dist_d_tl(epar.stim.dist_d_tl ~= 0);
    epar.stim.dist_d_tr = epar.stim.dist_d_tr(epar.stim.dist_d_tr ~= 0);

    % Columns represent different difficulty levels (1-16), rows represent
    % different positions of the gap (1 == bottom, 2 == left, 3 == right,
    % 4 == top)
    epar.stim.targ_e = [epar.stim.targ_e_b; epar.stim.targ_e_l;  ...
                        epar.stim.targ_e_r; epar.stim.targ_e_t]; % Targets
    epar.stim.targ_d = [epar.stim.targ_d_b; epar.stim.targ_d_l;  ...
                        epar.stim.targ_d_r; epar.stim.targ_d_t];
    epar.stim.dist_e = [epar.stim.dist_e_bl; epar.stim.dist_e_br;  ...
                        epar.stim.dist_e_tl; epar.stim.dist_e_tr]; % Distractors
    epar.stim.dist_d = [epar.stim.dist_d_bl; epar.stim.dist_d_br;  ...
                        epar.stim.dist_d_tl; epar.stim.dist_d_tr];


    %% Go back to root, when finished with everything
    cd('../')

end