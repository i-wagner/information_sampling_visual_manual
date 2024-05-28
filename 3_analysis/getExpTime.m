function availableTimeAfterExclusion = getExpTime(availableTime, lostTime, tempUnit)

    % Correct time, available to complete trials, for the time lost due to
    % trial exclusions
    %
    % Input
    % availableTime:
    % double; available time to complete trials. HAS TO BE IN SECONDS
    %
    % lostTime:
    % matrix; how much time participants lost due to trial exclusions 
    %
    % tempUnit:
    % string; unit in which "lostTime" is provided. Either "ms" or "s"
    % 
    % Output
    % availableTimeAfterExclusion:
    % matrix; availableTime corrected by lostTime, seperately for each
    % matrix entry

    %% Account for time lost due to trial exclusions
    assert(any(strcmp(tempUnit, ["ms", "s"])), ...
           "Please use valid unit-flag for input!");

    if strcmp(tempUnit, "ms")
        lostTime = lostTime ./ 60;
    end
    availableTimeAfterExclusion = availableTime - lostTime;

end