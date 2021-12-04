function img = preprocessStimuli(img, contrastAdjustmentFactor)

    %% Adjust the contrast of square element, placed within the stimulus
    for c = 1:size(img, 3) % Color channel

        % Get color values of the rectangular element
        img2 = [];
        img2 = img(12:38, 12:38, c);

        % Process
        img2 = double(img2); % Convert values to double
        img2 = (img2 - 128) .* contrastAdjustmentFactor; % Adjust contrast by factor
        img2 = round(img2 + 128); % Transform back
        img2 = uint8(img2);

        % Create output
        img(12:38, 12:38, c) = img2;

    end

end