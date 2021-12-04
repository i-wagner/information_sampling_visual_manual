%CODER.HARDWAREBASE Define an interface to target hardware
%
%   CODER.HARDWAREBASE is an abstract class which defines the interface
%   methods for using target hardware with MATLAB Coder.
%
%   The following methods must be implemented by classes which derive from
%   this base class.
%
%   addedToConfig(OBJ, CFG) Called when this object OBJ has been added to a
%   MATLAB Coder configuration object. A handle to the configuration object
%   is passed as the CFG parameter.
%
%   preBuild(OBJ, CFG) Called by the CODEGEN command prior to compiling the
%   source MATLAB code. A handle to the configuration object to be used
%   during compilation is passed as the CFG parameter.
%
%   postCodegen(OBJ, CFG, BUILDINFO) Called by the CODEGEN command after
%   converting the source MATLAB code to C/C++ code, but before compiling
%   the generated C/C++ code. A handle to the configuration object used
%   during compilation is passed as the CFG parameter. A handle to the
%   RTW.BuildInfo object used during compilation is passed as the BUILDINFO
%   parameter.
%
%   postBuild(OBJ, CFG, BUILDINFO) Called by the CODEGEN command after
%   generating and compiling the C/C++ code. A handle to the configuration
%   object used during compilation is passed as the CFG parameter. A handle
%   to the RTW.BuildInfo object used during compilation is passed as the
%   BUILDINFO parameter.
%
%   errorHandler(OBJ, CFG) Called by the CODEGEN command if an error is
%   detected while executing the command. A handle to the configuration
%   object used during compilation is passed as the CFG parameter.
%
%   TD = getCoderHardwareSettings(OBJ) Called by the CODEGEN command when
%   any compiled MATLAB source code invokes the coder.hardwareSettings
%   function. This function should return a structure containing fields
%   that represent the user-accesssible properties of this object.
%
%   See also coder.config, coder.hardwareSettings.

%   Copyright 2012-2019 The MathWorks, Inc.

classdef HardwareBase < dynamicprops
    methods (Access=public, Abstract)
        addedToConfig(obj, cfg);
        preBuild(obj, cfg);
        postCodegen(obj, cfg, buildInfo);
        postBuild(obj, cfg, buildInfo);
        errorHandler(obj, cfg);
        
        s = getCoderHardwareSetting(obj, property);
    end
end
