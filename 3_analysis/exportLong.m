function exportLong(exper, colNames, filePath, data, varargin)

    % Exports data in long format, i.e., rows are observations
    %
    % NOTE 1:
    % Before export, excluded subjects are dropped from input matrix
    % 
    % NOTE 2:
    % "data" input has to be in short-format, i.e., rows have to be
    % participants, columns dependent variables. The order of dependent
    % variables in "data" has to match the order of factor levels in
    % "varargin"
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % colNames:
    % string array; column names for exported data
    %
    % filePath:
    % string; export path
    %
    % data:
    % matrix; data to export
    %
    % varargin:
    % matrix; indinces for factor levels of dependent variable.
    %
    % Output
    % --

    %% Export data
    nFactors = numel(varargin);

    dataLong = [];
    for s = 1:exper.n.SUBJECTS % Subject
        thisSubject.number = exper.num.SUBJECTS(s);
        thisSubject.dv = data(s,:)';
        thisSubject.nDatapoints = numel(thisSubject.dv);

        % Build data matrix in long format
        thisSubject.dvLong = ...
            zeros(thisSubject.nDatapoints,1) + thisSubject.number;
        for f = 1:nFactors % Factor
            thisSubject.dvLong = [thisSubject.dvLong, varargin{f}];
        end
        thisSubject.dvLong = [thisSubject.dvLong, thisSubject.dv];
        if all(isnan(thisSubject.dv))
            thisSubject.dvLong = NaN(size(thisSubject.dvLong));
        end

        dataLong = [dataLong; thisSubject.dvLong];
    end

    isValidSubject = all(~isnan(dataLong), 2);
    dataLong = dataLong(isValidSubject,:);
    dataLong = array2table(dataLong);
    dataLong.Properties.VariableNames = colNames;
    writetable(dataLong, filePath)

end