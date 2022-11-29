function dev = lossFunction(freeParameter, gain, propChoicesEasy_emp, fixChosen_emp, fixEasy_emp, setSizes_emp)

%     [propChoicesEasy_pred, propFixations_pred] = test_unsystematic_fixations_2(setSizes_emp, gain, freeParameter);
    [propChoicesEasy_pred, propFixations_pred] = test_unsystematic_fixations_3(setSizes_emp, gain, freeParameter);

    propFixationsChosen_pred = propFixations_pred(:, 1, 2) ./ propFixations_pred(:, 3, 2);
    propFixationsEasy_pred    = propFixations_pred(:, 1, 1) ./ propFixations_pred(:, 3, 1);

%     dev = sum((propChoicesEasy_emp - propChoicesEasy_pred(:, 1)).^2, 'omitnan');
%     dev = sum((fixChosen_emp' - propFixationsChosen_pred).^2, 'omitnan');
    dev = sum(([propChoicesEasy_emp; fixChosen_emp'] - [propChoicesEasy_pred(:, 1); propFixationsChosen_pred]).^2, 'omitnan');
%     dev = sum(([propChoicesEasy_emp; fixChosen_emp'; fixEasy_emp'] - [propChoicesEasy_pred(:, 1); propFixationsChosen_pred; propFixationsEasy_pred]).^2, 'omitnan');

end