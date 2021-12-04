function [valid, error] = checkHardwareGuiCompliance(varargin)
    %CHECKHARDWAREGUICOMPLIANCE Checks the compliance of hardware targets 
    %   with the MATLAB Coder app. Compliance guarantees that targets will
    %   be visible and processable by the app but not necessarily that they
    %   will behave exactly as intended.
    %
    %   [VALID,ERROR] = coder.internal.checkHardwareGuiCompliance(HARDWARE)
    %   returns whether or not the given HARDWARE is compliant with the
    %   MATLAB Coder app's requirements for hardware specification and the
    %   validation errors if the hardware is invalid. The HARDWARE input 
    %   can be either a coder.Hardware object or a target's name.
    %
    %   [VALID,ERROR] = coder.internal.checkHardwareGuiCompliance(h, h2, ..., hN)
    %   returns the validity and validation errors for multiple hardware inputs 
    %   as a logical vector and a one-dimensional cell array, respectively.
    %   Each hardware input may be either a coder.Hardware object or the
    %   name of a target.
    %
    %   [VALID,ERROR] = coder.internal.checkHardwareGuiCompliance() returns
    %   a logical vector indicating the validity of all available targets 
    %   and a cell array indicating the reasons for noncompliance.  
    
    % Copyright 2015-2019 The MathWorks, Inc.
    
    
    if nargin == 1
        [valid, error] = validateHardware(normalizeHardware(varargin{1}));
        return;
    elseif nargin == 0
        hardwareObjects = emlcprivate('projectCoderHardware');
    else
        hardwareObjects = varargin;
    end
    
    valid = false(size(hardwareObjects));
    error = cell(size(hardwareObjects));
    
    for i = 1:numel(hardwareObjects)
        [valid(i), error{i}] = validateHardware(normalizeHardware(hardwareObjects{i}));        
    end
    
    %% -------------------------------------------------------------------
    function hwArg = normalizeHardware(hwArg)
        if ischar(hwArg)
            % Treat as hardware name
            hwArg = emlcprivate('projectCoderHardware', hwArg);
        end
    end
end

%% =======================================================================
function [valid, err] = validateHardware(hardware) 
    persistent validProdDeviceTypes;
    if isempty(validProdDeviceTypes)
        validProdDeviceTypes = generateProdDeviceTypes();
    end
    
    if isempty(hardware)
        valid = true;
        err = [];
        return;
    end
    
    try 
        validateattributes(hardware, {'coder.Hardware'}, {});
        
        % Verify that all needed properties exist on the coder.Hardware
        % object.
        validateattributes(hardware.Name, {'char'}, {'nonempty'});
        validateattributes(hardware.Version, {'char', 'double'}, {'nonempty'});
        validateattributes(hardware.HardwareInfo, {'codertarget.targethardware.TargetHardwareInfo'}, {'size', [1 1]});
        validateattributes(hardware.ParameterInfo, {'struct'}, {'size', [1 1]});
        
        % Validate the HardwareInfo struct.
        verifyFieldsAndProperties(hardware.HardwareInfo, 'Name', 'TargetName', 'DeviceID', ...
            'SubFamily', 'TargetFolder', 'ProdHWDeviceType', 'ToolChainInfo');
        verifyFieldsAndProperties(hardware.HardwareInfo.ToolChainInfo, 'Name', 'LoaderName', ...
            'LoadCommand', 'IsLoadCommandMATLABFcn');
        assert(ismember(lower(hardware.HardwareInfo.ProdHWDeviceType), validProdDeviceTypes), ...
            '"%s" is not a valid ProdHWDeviceType', hardware.HardwareInfo.ProdHWDeviceType);

        % Validate the parameter specifications.
        verifyFieldsAndProperties(hardware.ParameterInfo, 'ParameterGroups', 'Parameter');
        cellfun(@(paramGroup) assert(ischar(paramGroup)), hardware.ParameterInfo.ParameterGroups);
        % There should be a Parameter group specification for each
        % ParameterGroup indicated.
        validateattributes(hardware.ParameterInfo.Parameter, {'cell'}, ...
            {'size', size(hardware.ParameterInfo.ParameterGroups)});
        % Validate each individual parameter specification.
        validateParameters(hardware.ParameterInfo.Parameter);        
        
        valid = true;
        err = [];
    catch me
        err = me;
        valid = false;
    end     
end

%% =======================================================================
function validateParameters(parameters)
    validateattributes(parameters, {'cell'}, {});
    
    if ~usejava('jvm')
        return;  % No JVM, no GUI, can't do any further validation
    end
    
    % Gather the names of all the parameter fields that the GUI cares about.
    persistent javaParamFields;
    if isempty(javaParamFields)
        javaParamFields = javaMethod('getRequiredFields', 'com.mathworks.toolbox.coder.target.CtRawField');
        javaParamFields = cell(javaParamFields.toArray());
        for i = 1:numel(javaParamFields)
            javaParamField = javaParamFields{i};
            javaParamFields{i} = char(javaParamField.getRawKey());
        end
    end
    
    for i = 1:numel(parameters)
        % Validate each parameters group.
        cellfun(@(param) verifyFieldsAndProperties(param, javaParamFields{:}), parameters{i});
    end  
end

%% =======================================================================
function verifyFieldsAndProperties(actual, varargin)
    assert(isstruct(actual) || isobject(actual));    
    if isstruct(actual)
        fieldsOrProps = fields(actual);
    else
        fieldsOrProps = properties(actual);
    end    
    assert(all(ismember(varargin, fieldsOrProps)));
end

%% =======================================================================
function deviceTypes = generateProdDeviceTypes()
    hwImpl = coder.HardwareImplementation();
    prodVendors = emlcprivate('getHardwareVendorNames', 'Production');
    deviceTypes = cell(size(prodVendors));
    for i = 1:numel(prodVendors)
        typeNames = emlcprivate('getHardwareTypeNames', prodVendors{i}, hwImpl);
        deviceTypes{i} = cell(1, numel(typeNames));
        for j = 1:numel(typeNames)
            deviceTypes{i}{j} = sprintf('%s->%s', prodVendors{i}, typeNames{j});
        end
    end
    deviceTypes = lower([deviceTypes{:}]);
end
