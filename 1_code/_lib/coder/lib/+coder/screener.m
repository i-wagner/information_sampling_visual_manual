function varargout = screener(varargin)
%CODER.SCREENER Determine if a function is suitable for code generation 
% 
%   CODER.SCREENER(FCNS) provides a quick approximate analysis of 
%   one or more functions (FCNS) identifying MATLAB Coder compliance 
%   issues. It displays a report with the findings. 
% 
%   The inputs should be entry point functions. Any other non-MathWorks 
%   functions invoked directly or indirectly by these will be discovered 
%   and analyzed also. 
% 
%   CODER.SCREENER will not analyze MathWorks functions. 
% 
%   Example: 
%      coder.screener('myfcn'); 
%      coder.screener('entrypoint1', 'entrypoint2'); 
% 
% See also codegen. 

%   Copyright 2009-2019 The MathWorks, Inc.

if nargin > 1 && strcmp(varargin{1}, '-text')
    
    X = coderprivate.emlscreener_kernel(varargin{2:end});

    coderprivate.emlscreener_genreport(X);

    if nargout == 1
        varargout{1} = X;
    end

else
    if codergui.internal.isMatlabOnline()
        error(message('Coder:common:ScreneerMOError')); 
    end
    
    validExtensions = {'.m'};
    validExtErrorStr = '*.m';
    
    if com.mathworks.toolbox.coder.model.CoderFileSupport.isCoderMlxEnabled()
        validExtensions{end+1} = '.mlx';
        validExtErrorStr = '*.m, *.mlx';
    end
    
    files = java.util.ArrayList;
    
    screenerMode = 'C';
    
    for i = 1:nargin
        arg = varargin{i};
        
        if startsWith(arg, '-')
            switch arg
                case '-gpu'
                    screenerMode = 'GPU';
                    continue;
                case '-c'
                    screenerMode = 'C';
                    continue;
            end
        end
        
        filepath = which(char(arg));
        
        if ~exist(filepath, 'file')
            error(message('Coder:common:ProjectFileNotFound', arg));
        end
        
        [~, ~, fileExt] = fileparts(filepath);
        
        if ~any(strcmpi(fileExt, validExtensions))
            error(message('Coder:common:NonMFile', validExtErrorStr));
        end
        
        files.add(java.io.File(filepath));
    end
    
    if ~files.isEmpty()
        screenerMode = com.mathworks.toolbox.coder.screener.ScreenerTarget.(screenerMode);
        try
            com.mathworks.toolbox.coder.screener.ScreenerReportDialog.show(files, screenerMode);    
        catch e
            if isa(e.ExceptionObject, 'com.mathworks.toolbox.coder.screener.MathWorksFileException')
               error(message('Coder:common:ScreeningMathWorksFile')); 
            end
        end
    end
end

