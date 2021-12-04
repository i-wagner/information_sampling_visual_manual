function result = hasPublicStaticMethod(className, methodName)
%

%   Copyright 2019 The MathWorks, Inc.

metaClass = meta.class.fromName(className);
if isempty(metaClass)
    result = false;
    return
end
methodList = metaClass.MethodList;
methodList = methodList(strcmp({methodList.Name}, methodName));
if isempty(methodList)
    result = false;
    return
end
result = strcmp({methodList.Access}, 'public') && methodList.Static;
