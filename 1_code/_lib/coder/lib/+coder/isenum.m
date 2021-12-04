function y = isenum(A)
%CODER.ISENUM Determine if A is an enumeration.
%
%   Example:
%     MyEnum.m:
%     classdef(Enumeration) MyEnum < int32
%         enumeration
%             MyEnum.MYCONSTANT(100)
%             MyEnum.MYCONSTANT2(200)
%         end
%     end
%
%     foo.m:
%     function y = foo(x)
%     if (coder.isenum(x))
%         y = 'Is an enumeration object';
%     else
%         y = 'is NOT an enumeration object';
%     end
%
%   y = foo(MyEnum.CONSTANT(100))
%   y = foo(10)

%   Copyright 2008-2019 The MathWorks, Inc.

y = isenum(A);
