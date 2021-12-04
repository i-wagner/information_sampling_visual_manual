function [redirectedObj, changed] = toRedirected(obj)
    % toRedirected

    % Copyright 2016-2017 The MathWorks, Inc.
    objClass = builtin('class', obj);
    if issparse(obj)
        redirectedObjClass = 'coder.internal.sparse';
    else
        redirectedObjClass = coder.internal.getRedirectedClassName(objClass);
    end
    if strcmp(redirectedObjClass, objClass)
        redirectedObj = obj;
        changed = false;
    else
        if ~coder.internal.hasPublicStaticMethod(redirectedObjClass, 'matlabCodegenToRedirected')
                % If there is no matlabCodegenToRedirected then this class cannot
                % be accepted as an input.
            error(message('Coder:common:NoMatlabCodegenToRedirected', objClass));
        end
        redirectedObj = eval([redirectedObjClass '.matlabCodegenToRedirected(obj)']);
        changed = true;
    end
