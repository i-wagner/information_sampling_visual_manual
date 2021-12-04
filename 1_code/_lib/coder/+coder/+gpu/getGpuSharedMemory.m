function [sharedMemory] = getGpuSharedMemory(ctx)

%   Copyright 2019 The MathWorks, Inc.

    cfg = coder.gpu.getGpuConfig(ctx); 
    if ~isempty(cfg)
        sharedMemory = cfg.SharedMemorySize; 
    else
        defaultConfig = coder.gpu.config;
        sharedMemory = defaultConfig.SharedMemorySize;
    end
end
