function gapLocation = determineGapLocation(stimulusToShow, epar)
    
    %% Determine on which side of the target the gap is placed
    % Bottom?
    if ismember(stimulusToShow, epar.stim.targ_e_b) || ...
       ismember(stimulusToShow, epar.stim.targ_d_b)

        gapLocation = 1;

    % Top?
    elseif ismember(stimulusToShow, epar.stim.targ_e_t) || ...
           ismember(stimulusToShow, epar.stim.targ_d_t)

        gapLocation = 2;

    % Left?
    elseif ismember(stimulusToShow, epar.stim.targ_e_l) || ...
           ismember(stimulusToShow, epar.stim.targ_d_l)

        gapLocation = 3;

    % Right?
    elseif ismember(stimulusToShow, epar.stim.targ_e_r) || ...
           ismember(stimulusToShow, epar.stim.targ_d_r)

        gapLocation = 4;

    end

end