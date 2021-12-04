function [imgs, comp] = createTargDisComposite(imgNames, img_cueB, img_cueR, stimCont, bckgrdC)

    % Creates composite image of targets/distractors. The individual images
    % of the composite have gaps at both sides, which this script inserts
    % Input
    % imgNames: Images, we want to use for composite
    % img_cueB/img_cueR:
    % stimCont: Constant we use to reduce the contrast of the rectangular
    %           elements
    % bckgrdC: Color of background, on which we will place the stimuli
    % Output
    % comp: Matrix, containing the composite image

    %% Load images, preprocess and add second gap
    % Loop through images, add a second gap and store the edited images
    noImgs       = size(imgNames, 1);
    imgs_forComp = cell(1, noImgs);
    imgs         = cell(1, noImgs);
    idx_targets  = 3:4;
    for i = 1:noImgs % Image

        currentImg = imread(imgNames{i}); % Load image
        currentImg = preprocessStimuli(currentImg, stimCont); % Adjust contrast of rectangular element
        currentImg = double(currentImg); % Convert to double

        % Close the gap
        if ismember(i, idx_targets)

            if i == 5 % Vertical target

                currentImg(25, 29, :) = 0;

            else % Horizontal target

                currentImg(21, 25, :) = 0;

            end

        else

            if i == 1 % Northeast distractor

                currentImg(22, 28, :) = 35;
                currentImg(22, 27, :) = 46;
                currentImg(23, 28, :) = 47;

            else % Northwest distractor

                currentImg(22, 22, :) = 35;
                currentImg(22, 23, :) = 46;
                currentImg(23, 22, :) = 43;

            end

        end

        % Save preprocessed image and convert it to uint8
        imgs_forComp{i} = currentImg;
        imgs{i}         = uint8(currentImg);

    end
%     figure(1); for s = 1:noImgs; subplot(1, noImgs, s); imshow(uint8(imgs{s})); end % Plot edited images


    %% Create composite image of targets and distractors
    % Loop through images and create a composite of all target/distractor
    % orientations; for this, we just sum the left/right tilted distractor
    % and the horizontal/vertical target
    comp_pattern = uint8(((imgs_forComp{1}(12:38, 12:38, :) - bckgrdC) + (imgs_forComp{2}(12:38, 12:38, :) - bckgrdC) + ...
                          (imgs_forComp{3}(12:38, 12:38, :) - bckgrdC) + (imgs_forComp{4}(12:38, 12:38, :) - bckgrdC)) ...
                         + bckgrdC);

    img_cueB(12:38, 12:38, :) = comp_pattern;
    img_cueR(12:38, 12:38, :) = comp_pattern;

    comp{1} = img_cueB;
    comp{2} = img_cueR;

%     figure(2); for s = 1:2; subplot(1, 2, s); imshow(comp{s}); end % Plot composite images

end