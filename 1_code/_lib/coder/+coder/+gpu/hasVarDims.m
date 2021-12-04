function flag = hasVarDims(var)
%#codegen

flag = ~coder.internal.isConst(size(var));
