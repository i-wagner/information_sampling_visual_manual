function [] = updateGpuBuildInfo(bldParams, codingTarget)
%

%   Copyright 2017-2019 The MathWorks, Inc.

gpuCfg = bldParams.configInfo.GpuConfig;
if strcmp(codingTarget, 'mex')
    isMexBuild = true;
else
    isMexBuild = false;
end

% Undef printf in device code for MEX target
if isMexBuild
    customCode = bldParams.configInfo.CustomHeaderCode;
    undefPrintf = sprintf('\n#ifdef __CUDA_ARCH__ \n#undef printf \n#endif\n');
    bldParams.configInfo.CustomHeaderCode = [undefPrintf customCode];
end

% If using custom compute capability flag, pass that directly to
%   the GPU compiler. Otherwise modify the '#.#' formatted compute
%   compatibility value into a sm_## compiler flag.
if (~isempty(gpuCfg.CustomComputeCapability))
    gpuCompFlags = gpuCfg.CustomComputeCapability;
    ptxVersion = '350';
else
    ccVals = regexp(gpuCfg.ComputeCapability,'\.','split');
    gpuCompFlags = ['-arch sm_' ccVals{1} ccVals{2} ' '];
    ptxVersion = [ccVals{1} ccVals{2} '0'];
end

bldParams.buildInfo.addLinkFlags({gpuCompFlags});
bldParams.buildInfo.addDefines(['-DMW_CUDA_ARCH=' ptxVersion]);

if (~isempty(gpuCfg.CompilerFlags))
    gpuCompFlags = [gpuCompFlags ' ' gpuCfg.CompilerFlags];
end

cfgCudaVer = gpuCfg.CudaVersion;
useShippingLibs = gpuCfg.UseShippingLibs;

cudaVerFlags = doCudaVersionChecksAndGetFlags(isMexBuild, cfgCudaVer, useShippingLibs);

if ~isempty(cudaVerFlags)
    gpuCompFlags = [gpuCompFlags ' ' cudaVerFlags];
end

bldParams.buildInfo.addCompileFlags({gpuCompFlags}, {'CU_OPTS'});

if gpuCfg.EnableRuntimeLog
    bldParams.buildInfo.addDefines('-DMW_GPUCODER_RUNTIME_LOG');
end

end

function cudaVerFlags = doCudaVersionChecksAndGetFlags(isMexBuild, cfgCudaVer, useShippingLibs)

[defCudaVer, nvccFound, nvccCudaVer] = getCudaVersion;

if ~useShippingLibs
    if isMexBuild
        % check if gpucfg's CudaVersion (hidden) matches default CUDA Version for MEX
        if ~isempty(cfgCudaVer) && ~strcmp(cfgCudaVer, defCudaVer)
            error(message('gpucoder:common:InvalidToolkitVersionConfig', cfgCudaVer, defCudaVer));
        end
        
        % check if nvcc CudaVersion matches default CUDA Version for MEX
        if nvccFound && ~isempty(nvccCudaVer) && ~strcmp(nvccCudaVer, defCudaVer)
            error(message('gpucoder:common:InvalidToolkitVersionNVCC', nvccCudaVer, defCudaVer));
        end
    else
        % Throw warning if nvcc CudaVersion is less than CUDA 9/10
        minCudaVer = '9';
        if nvccFound && ~isempty(nvccCudaVer) && str2double(nvccCudaVer) < str2double(minCudaVer)
            oldstate = warning('off', 'backtrace');
            warning(message('gpucoder:common:OlderToolkitVersionWarning', nvccCudaVer, minCudaVer));
            warning(oldstate.state, 'backtrace');
        end
    end
end

% Get flags based on CUDA Version
if useShippingLibs || ~nvccFound || isempty(nvccCudaVer)
    cudaVerFlags = getFlagsForCudaVersion(defCudaVer);
else
    cudaVerFlags = getFlagsForCudaVersion(nvccCudaVer);
end

end
