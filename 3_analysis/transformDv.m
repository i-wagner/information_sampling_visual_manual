function dvTransformed = transformDv(exper, anal, dv, transformation)

    % Transforms some dependent variable to approximate a normal
    % distribution
    %
    % NOTE:
    % Different transformations are better suited for different dependent
    % variables:
    % - log: works if dependent variable has no zeros
    % - logit: works if dependent variable has no zeros or ones
    % - arcsin: works for proportion data
    %
    % Input
    % exper:
    % structure; general experiment settings, as returned by the
    % "settings_exper" script
    %
    % anal:
    % structure; various analysis settings, as returned by the
    % "settings_analysis" script
    %
    % dv:
    % matrix; dependent variable to transform; can either be a cell or
    % double matrix
    %
    % transformation:
    % string; which transformation to apply. Can be one of the following:
    % log, log2, log10, boxcox, sqrt, arcsin, logit
    %
    % Output
    % dvTransformed:
    % matrix; same as input, but transformed

    %% Check input
    assert(any(strcmp(transformation, ...
                      ["log", "log2", "log10", "boxcox", "sqrt", "arcsin", "logit"])));

    %% Transform dependent variable
    nConditions = size(dv, 2);
    if strcmp(class(dv), "cell")
        dvTransformed = cell(size(dv));
    elseif strcmp(class(dv), "double")
        dvTransformed = NaN(size(dv));
    end
    
    for c = 1:nConditions % Condition
        for s = 1:exper.n.SUBJECTS % Subject
            thisSubject.number = exper.num.SUBJECTS(s);
            thisSubject.dv = dv{thisSubject.number,c};
            if ismember(thisSubject.number, anal.excludedSubjects) | ...
               isempty(dv) | ...
               all(isnan(thisSubject.dv))
                continue
            end

            if strcmp(transformation, "log")
                thisSubject.dv = log(thisSubject.dv);
            elseif strcmp(transformation, "log2")
                thisSubject.dv = log2(thisSubject.dv);
            elseif strcmp(transformation, "log10")
                thisSubject.dv = log10(thisSubject.dv);
            elseif strcmp(transformation, "boxcox")
                thisSubject.dv = boxcox(thisSubject.dv);
            elseif strcmp(transformation, "sqrt")
                thisSubject.dv = sqrt(thisSubject.dv);
            elseif strcmp(transformation, "arcsin")
                % https://www.geeksforgeeks.org/how-to-perform-arcsine-transformation-in-r/
                thisSubject.dv = asin(sqrt(thisSubject.dv));
            elseif strcmp(transformation, "logit")
                % https://de.mathworks.com/matlabcentral/fileexchange/
                % 131509-logit-function-to-transform-proportional-data-
                % for-regression
                error([transformation, " not implemented!"]);
            end

            if strcmp(class(dv), "cell")
                dvTransformed{thisSubject.number,c} = thisSubject.dv;
            elseif strcmp(class(dv), "double")
                dvTransformed(thisSubject.number,c) = thisSubject.dv;
            end
            clear thisSubject
        end
    end

end