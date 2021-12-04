function [xCircS, yCircS] = defineAOIs(stimulusLocationOnX, stimulusLocationOnY, desiredAOIsize, stimSize, conversionFactorX)

    % Calculates x/y-coordiantes coordinates of AOIs of all stimuli,
    % displayed in a trial
    % stimulusLocationOnX: locations of stimuli on x-axis (one trial)
    % stimulusLocationOnY: locations of stimuli on y-axis (one trial)
    % stimSize: size of a stimulus (in pixel)
    % desiredAOIsize: overall size of AOI (diameter around center of a
    %                 stimulus; in deg)
    % conversionFactorX: conversion factor to transform pixel to deg for
    %                    x-axis
    % Output
    % xCircS2/yCircS2: row vector with x/y-coordinates of AOIs around all
    %                  stimuli, displayed in a trial; individual stimuli are
    %                  separated by a singular "NaN" entry

    %% Define AOI size
    stim_sizePx   = stimSize;                        % Stimulus size/diameter (in px)
    stim_diameter = stim_sizePx * conversionFactorX; % Stimulus size/diameter (in deg)

    % Define an AOI around the stimuli; a saccade landing in this AOI is 
    % defined as landing "on stimulus"
    stim_diameterAOI = stim_diameter + (desiredAOIsize - stim_diameter);
    stim_radiusAOI   = stim_diameterAOI / 2;


    %% Calculate AOI around all displayed stimuli
    stim_angles = 0:0.01:2*pi;
    xCircS      = [];
    yCircS      = [];
    for ss = 1:size(stimulusLocationOnX, 2)

        % We add a "NaN" at the end of the vector, so we can later separate
        % between the coordinates of different stimuli (necessary for "inpolygons")
        xCircS(ss, :) = [cos(stim_angles) .* stim_radiusAOI + stimulusLocationOnX(1, ss) NaN];
        yCircS(ss, :) = [sin(stim_angles) .* stim_radiusAOI + stimulusLocationOnY(1, ss) NaN];

    end

end