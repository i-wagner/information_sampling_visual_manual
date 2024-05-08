function out = getStats(data)

    % Calculates mean, median, standard deviation, range, and 95%
    % confidence interval of some vairable of interest
    %
    % Input
    % data: 
    % vector; data for which to calculate statistics
    %
    % Output
    % out: 
    % structure; statistics for variable of interest

    %% Get statistics
    out.mean = mean(data, 'omitnan');
    out.median = median(data, 'omitnan');
    out.std = std(data, 'omitnan');
    out.range = [min(data), max(data)];
    out.ci = ci_mean(data);

end