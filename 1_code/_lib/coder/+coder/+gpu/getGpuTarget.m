function [target] = getGpuTarget(ctx)
%

%   Copyright 2017 The MathWorks, Inc.

    target = '';
    if ((~isequal(ctx, [])) && (ctx.isCodeGenTarget({'rtw'})))
        try
            target = ctx.getConfigProp('Toolchain');
        catch
        end
    end
end
