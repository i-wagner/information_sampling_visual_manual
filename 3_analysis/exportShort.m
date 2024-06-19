function exportShort(data, colNames, filePath)

    % Exports data in short format, i.e., rows are participants, columns
    % are variables
    %
    % NOTE:
    % Before export, excluded subjects are dropped from input matrix
    %
    % Input
    % data:
    % matrix; data to export
    %
    % colNames:
    % string array; column names for exported data
    %
    % filePath:
    % string; export path
    %
    % Output
    % --

    %% Export data
    datForExport = data;

    idxKeep = all(~isnan(datForExport), 2);
    datForExport = datForExport(idxKeep,:);
    datForExport = array2table(datForExport);
    datForExport.Properties.VariableNames = colNames;
    writetable(datForExport, filePath);

end