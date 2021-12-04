function [status, errorMsg] = validateGpuDriver()
    status = true;
    errorMsg = '';
    try 
        gpuDevice;
    catch E
        errorId = E.identifier;
        switch errorId
            case {'parallel:gpu:device:CouldNotLoadDriver',...
                    'parallel:gpu:device:DriverLoadFailed',...
                    'parallel:gpu:device:UnknownOldDriver',...
                    'parallel:gpu:device:FailedToQueryVersion',...
                    'parallel:gpu:device:OldDriver'}
                status = false;
                errorMsg = ['Acceleration using GPUs is not available. ', newline, E.message];
            otherwise
                % No driver issue.
        end
    end
end