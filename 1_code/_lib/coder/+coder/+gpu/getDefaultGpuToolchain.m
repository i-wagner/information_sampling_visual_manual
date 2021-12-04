function selectedTC = getDefaultGpuToolchain(targetLang, isCodingMex)
%getDefaultGpuToolchain : This function selects the appropriate default
% toolchain for GPU coder when none is selected by the user

selectedTC = '';

if ispc
    % On Windows, because NVCC only supports visual studio, we need to 
    %  find an appropriate version. First, check mex to see if a valid
    %  version of MSVC is selected
    supportedTCMap = coder.gpu.getSupportedTCMap;
    cc = mex.getCompilerConfigurations(targetLang,'Selected');

    if (~isempty(cc) && isKey(supportedTCMap,cc.ShortName))
        % Check if a supported MSVC toolchain is specified by mex -setup.
        % Assume this is the user's preference, and use the corresponding
        % NVCC toolchain
        selectedTC = supportedTCMap(cc.ShortName);
    elseif (isCodingMex)
        % If no valid toolchain is found and we are coding for MEX, throw
        % an error
        if (~isempty(cc))
            compName = cc.Name;
        else
            compName = 'NONE';
        end
        warning(message('gpucoder:common:NoValidMexGpuCompiler',compName));
    else
        % If no supported toolchain is found and we are coding RTW, look
        % for the latest version of MSVC present on the host machine
        if (~isempty(getenv('VS160COMNTOOLS')))
            selectedTC = supportedTCMap('MSVC160');
        elseif (~isempty(getenv('VS150COMNTOOLS')))
            selectedTC = supportedTCMap('MSVC150');
        elseif (~isempty(getenv('VS140COMNTOOLS')))
            selectedTC = supportedTCMap('MSVC140');
        elseif (~isempty(getenv('VS120COMNTOOLS')))
            selectedTC = supportedTCMap('MSVC120');
        else
            selectedTC = 'Automatically locate an installed toolchain';
            % No valid toolchain found on the host
            warning(message('gpucoder:common:NoValidGpuCoderToolchain'));
        end
    end
else
    selectedTC = 'NVIDIA CUDA | gmake (64-bit Linux)';
end
                    
end