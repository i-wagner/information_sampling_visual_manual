function hitrates = infSampling_propCorrect(hitsMisses, targetInTrial)

    % Calculate a participants perceptual performance
    % Input
    % hitsMisses:    vector with indicators if a trial was a hit or a miss
    % targetInTrial: vector with indicators, which target was chosen
    % Output
    % hitrates:      column-vector with overall and target-specific hitrates
    %                (1): overall
    %                (2): easy target
    %                (3): hard target

    %% Drop NaNs (i.e., excluded trials)
    % Excluded trials are determined by checking for NaN entries in the
    % vector that stores the chosen target; we do this, because this vector
    % also includes trials in which no chosen target could be determined
    % (i.e., trials in which the last fixated AOI did not contain any
    % target)
    li_excld = isnan(targetInTrial);

    hitsMisses    = hitsMisses(~li_excld);
    targetInTrial = targetInTrial(~li_excld);
    clear li_excld


    %% Calculate overall hitrate and hitrate seperate for both target difficults
    hitrates = NaN(3, 1);

    hitrates(1, 1) = sum(hitsMisses == 1) / numel(hitsMisses);             % Overall
    for d = 1:2 % Target difficulty

        hitrates(d+1, 1) = sum(hitsMisses == 1 & targetInTrial == d) / ... % Easy/hard target
                           sum(targetInTrial == d);

    end

end