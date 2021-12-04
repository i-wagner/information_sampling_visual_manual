function redirectedClass = getRedirectedClassName(cls)
    % getRedirectedClassName

    % Copyright 2016-2019 The MathWorks, Inc.

    if strcmp(cls, 'string')
        redirectedClass = 'coder.internal.string';
    elseif coder.internal.hasPublicStaticMethod(cls, 'matlabCodegenRedirect')
        f = eval(['@' cls '.matlabCodegenRedirect']);
        if ~nargin(f)
            error(message('Coder:common:MatlabCodegenRedirectMustAcceptOneInput', cls));
        end
        redirectedClass = feval(f, coder.target);
    else
        redirectedClass = cls;
    end
