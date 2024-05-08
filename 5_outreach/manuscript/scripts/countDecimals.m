function nDecimals = countDecimals(num)

    % Counts how many numbers occur after decimal point
    %
    % Input
    % num:
    % double; number to check for n decimals
    %
    % Output
    % nDecimals:
    % integer; n decimals in "num"

    %% Count decimals
    decimal = extractAfter(string(num), '.');
    nDecimals = 0;
    if ~ismissing(decimal)
        nDecimals = numel(char(decimal));
    end

end