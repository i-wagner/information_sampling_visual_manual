function showStats(stats, metric, exp, cond, nDecimals)

    % Formats and displays statistics of some variable of interest in the
    % command window
    %
    % Input
    % stats: 
    % structure; statistics of some output variable, as returned by the
    % "getStats" function
    %
    % metric: 
    % string; name of the metric/measure for which statistics where
    % calculated
    %
    % exp: 
    % string; experiment, in which metric/measure was recorded
    %
    % cond: 
    % string; condition, in which metric/measure was recorded
    %
    % nDecimals: 
    % integer; number of decimals to round to for output
    %
    % Output
    % --

    %% Format and show statistics
    disp(' ');
    disp(['Stats for ', metric, ' (', exp, ' experiment, ', cond, ...
          ' condition):']);
    disp(['Median = ', num2str(round(stats.median, nDecimals))]);
    disp(['M = ', num2str(round(stats.mean, nDecimals))]);
    disp(['STD = ', num2str(round(stats.std, nDecimals))]);
    disp(['Range = [', num2str(round(stats.range(1), nDecimals)), ...
          ', ', num2str(round(stats.range(2), nDecimals)), ']']);
    disp(['CI = [', num2str(round(stats.mean - stats.ci, nDecimals)), ...
          ', ', num2str(round(stats.mean + stats.ci, nDecimals)), ']']);

end