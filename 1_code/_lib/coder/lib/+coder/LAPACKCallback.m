%CODER.LAPACKCallback An abstract class for LAPACK callback

%   Copyright 2015-2019 The MathWorks, Inc.

classdef (Abstract) LAPACKCallback
    methods (Static, Abstract)
         headerName = getHeaderFilename()
         updateBuildInfo(aBuildInfo, context)
    end

    methods (Static)
        function [headerName] = getGpuHeaderFilename()
            headerName = 'cusolverDn.h';
        end
    end

end
