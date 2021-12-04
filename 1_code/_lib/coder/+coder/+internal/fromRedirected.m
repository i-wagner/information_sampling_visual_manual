function [obj, changed] = fromRedirected(redirectedObj)
redirectedObjClass = builtin('class', redirectedObj);
if coder.internal.hasPublicStaticMethod(redirectedObjClass, 'matlabCodegenFromRedirected')
    obj = eval([redirectedObjClass '.matlabCodegenFromRedirected(redirectedObj)']);
    changed = true;
else
    obj = redirectedObj;
    changed = false;
end
