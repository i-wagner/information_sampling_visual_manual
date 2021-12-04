function pstruct = parseParameterInputsML(parms,options,args)
%MATLAB Code Generation Private Function

%   Version of parseParameterInputs to be executed in MATLAB.
%   Note that instead of a varargin list, this version requires a third
%   input, and it must be a cell array.

%   Copyright 2009-2019 The MathWorks, Inc.
%#codegen


narginchk(3,3);
if isstruct(parms)
    pstruct = parseParameterInputsML(fieldnames(parms),options,args);
    return
end
coder.internal.assert(iscell(parms), ...
    'Coder:toolbox:eml_parse_parameter_inputs_2');
[caseSensitive,partialMatch,expandStructs,ignoreNulls] = ...
    processOptions(options);
[isMatch,isExactMatch] = parmMatchFunctions(caseSensitive,partialMatch);
nargs = numel(args);
% These are technical limitations of this implementation, so we check them
% here, regardless of whether another limitation may make them impossible
% to violate.
coder.internal.assert(nargs <= 65535, ...
    'Coder:toolbox:eml_parse_parameter_inputs_3');
nparms = length(parms);
coder.internal.assert(nparms <= 65535, ...
    'Coder:toolbox:eml_parse_parameter_inputs_4');
% Create and initialize the output structure.
pstruct = makeStruct(parms);
if nargs > 0
    % Parse VARARGIN{:}.
    t = inputTypes(args);
    if t(nargs) == 'n'
        % Last input cannot be a parameter name.
        error(message('Coder:toolbox:eml_parse_parameter_inputs_6', ...
            args{nargs}));
    end
    for k = 1:nargs
        if t(k) == 'n' % name
            % Find the index of the field args{k} in PARMS.
            pidx = findParm(args{k},parms, ...
                isMatch,isExactMatch,caseSensitive,partialMatch);
            % The parameter value is in args{k+1}. Set the value of
            % the field in PARMS accordingly.
            if ~ignoreNulls || ~isnull(args{k+1})
                pstruct.(parms{pidx}) = uint32(k+1);
            end
        elseif expandStructs && t(k) == 's' % expand structure
            opStructfieldNames = fieldnames(args{k});
            nOpStructFields = length(opStructfieldNames);
            for fidx = 1:nOpStructFields
                fname = opStructfieldNames{fidx};
                if ~ignoreNulls || ~isnull(args{k}.(fname))
                    % Find the index of the corresponding field in PARMS.
                    pidx = findParm(fname,parms, ...
                        isMatch,isExactMatch,caseSensitive,partialMatch);
                    % The parameter value is in the struct args{k} at
                    % field index fieldidx. Set the value of the field
                    % in PARMS accordingly.
                    pstruct.(parms{pidx}) = ...
                        combineIndices(uint32(k),uint32(fidx-1));
                end
            end
        else
            coder.internal.assert(t(k) == 'v', ...
                'Coder:toolbox:eml_parse_parameter_inputs_8');
        end
    end
end

%--------------------------------------------------------------------------

function [caseSensative,partialMatch,expandStructs,ignoreNulls] = ...
    processOptions(options)
% Extract parse options from options input structure, supplying default
% values if needed.
% Set defaults.
caseSensative = false;
expandStructs = true;
partialMatch = 'n'; % none
ignoreNulls = false;
% Read options.
if ~isempty(options)
    coder.internal.assert(isstruct(options), ...
        'Coder:toolbox:eml_parse_parameter_inputs_10');
    fnames = fieldnames(options);
    nfields = length(fnames);
    coder.unroll;
    for k = 1:nfields
        fname = fnames{k};
        if strcmp(fname,'CaseSensitivity')
            coder.internal.assert(isscalar(options.CaseSensitivity) && ...
                islogical(options.CaseSensitivity), ...
                'Coder:toolbox:eml_parse_parameter_inputs_11');
            caseSensative = options.CaseSensitivity;
        elseif strcmp(fname,'StructExpand')
            coder.internal.assert(isscalar(options.StructExpand) && ...
                islogical(options.StructExpand), ...
                'Coder:toolbox:eml_parse_parameter_inputs_12');
            expandStructs = options.StructExpand;
        elseif strcmp(fname,'PartialMatching')
            isfirst = strcmp(options.PartialMatching,'first') || ( ...
                isscalar(options.PartialMatching) && ...
                options.PartialMatching ~= false);
            isnone = strcmp(options.PartialMatching,'none') || ( ...
                isscalar(options.PartialMatching) && ...
                options.PartialMatching == false);
            isunique = strcmp(options.PartialMatching,'unique');
            coder.internal.assert(isfirst || isnone || isunique, ...
                'Coder:toolbox:eml_parse_parameter_inputs_13');
            if isunique
                partialMatch = 'u'; % unique
            elseif isfirst
                partialMatch = 'f'; % first
            else
                partialMatch = 'n'; % none
            end
        elseif strcmp(fname,'IgnoreNulls')
            coder.internal.assert(isscalar(options.IgnoreNulls) && ...
                islogical(options.IgnoreNulls), ...
                'Coder:toolbox:BadIgnoreNulls');
            ignoreNulls = options.IgnoreNulls;
        else
            error(message('Coder:toolbox:eml_parse_parameter_inputs_14'));
        end
    end
end

%--------------------------------------------------------------------------

function t = inputTypes(args)
% Returns an array indicating the classification of each argument as a
% parameter name, parameter value, option structure, or unrecognized. The
% return value must be constant folded.
nargs = numel(args);
t = blanks(nargs);
isval = false;
coder.unroll;
for k = 1:nargs
    if isval
        t(k) = 'v'; % value
        isval = false;
    elseif (ischar(args{k}) && isrow(args{k})) || ...
            (isstring(args{k}) && isscalar(args{k}))
        t(k) = 'n'; % name
        isval = true;
    elseif isstruct(args{k})
        t(k) = 's'; % structure
        isval = false;
    else
        t(k) = 'u'; % unrecognized
        isval = false;
    end
end

%--------------------------------------------------------------------------

function n = findParm(parm,parms,matchFun,exactMatchFun, ...
    caseSensitive,partialMatch)
% Find the index of parm in the parms list. Asserts if parm is not found.
[n,ncandidates] = findParmKernel(parm,parms,matchFun,exactMatchFun, ...
    partialMatch);
coder.internal.assert(ncandidates ~= 0, ...
    'Coder:toolbox:eml_parse_parameter_inputs_16',parm);
if ncandidates ~= 1
    coder.internal.assert(false, ...
        'Coder:toolbox:AmbiguousPartialMatch',parm, ...
        coder.internal.partialParameterMatchString( ...
        coder.internal.toCharIfString(parm),parms,caseSensitive));
end

%--------------------------------------------------------------------------

function [n,ncandidates] = findParmKernel(parm,parms, ...
    isMatch,isExactMatch,partialMatch)
uPartMatch = partialMatch == 'u'; % unique partial matching
n = 0;
ncandidates = 0;
nparms = length(parms);
for j = 1:nparms
    if isMatch(parms{j},parm)
        if isExactMatch(parms{j},parm)
            % An exact match rules out all other candidates.
            n = j;
            ncandidates = 1;
            break
        elseif uPartMatch || n == 0
            n = j;
            ncandidates = ncandidates + 1;
        else
            % In this case, partialMatch == 'f', we have a first partial
            % match in hand, and we are only scanning through the rest of
            % the parameters looking for an exact match. Consequently, this
            % partial match is ignored.
        end
    end
end

%--------------------------------------------------------------------------

function [isMatch,isExactMatch] = parmMatchFunctions(casesens,prtmatch)
% Return function handles for matching of parameter names. The matchFun
% respects the casesens and prtmatch options. The exactMatchFun replaces
% prtmatch with PM_NONE (no partial matching). Note that if prtmatch is
% PM_NONE to begin with, exactMatchFun simply returns true without doing
% any work, since it is meant to be used only after detecting a match with
% matchFun.
partial = prtmatch ~= 'n';
if casesens
    if partial
        isMatch = @isCaseSensitivePartialMatch;
        isExactMatch = @isCaseSensitiveMatch;
    else
        isMatch = @isCaseSensitiveMatch;
        isExactMatch = @returnTrue;
    end
else
    if partial
        isMatch = @isCaseInsensitivePartialMatch;
        isExactMatch = @isCaseInsensitiveMatch;
    else
        isMatch = @isCaseInsensitiveMatch;
        isExactMatch = @returnTrue;
    end
end

function p = isCaseSensitivePartialMatch(mstrparm,userparm)
if isempty(userparm)
    p = false;
else
    p = strncmp(mstrparm,userparm,strlength(userparm));
end

function p = isCaseInsensitivePartialMatch(mstrparm,userparm)
if isempty(userparm)
    p = false;
else
    p = strncmpi(mstrparm,userparm,strlength(userparm));
end

function p = isCaseSensitiveMatch(mstrparm,userparm)
if isempty(userparm)
    p = false;
else
    p = strcmp(mstrparm,userparm);
end

function p = isCaseInsensitiveMatch(mstrparm,userparm)
if isempty(userparm)
    p = false;
else
    p = strcmpi(mstrparm,userparm);
end

function p = returnTrue(~,~)
p = true;

%--------------------------------------------------------------------------

function n = combineIndices(vargidx,stfldidx)
% Returns a 'uint32'. Stores the struct field index (zero-based) in the
% low bits and the varargin index in the low bits.
% n = (struct_field_ordinal << 16) + vargidx;
n = bitor(bitshift(vargidx,16),stfldidx);

%--------------------------------------------------------------------------

function p = isnull(x)
% Returns true if x is [] and fixed-size.
p = isa(x,'double') && isequal(size(x),[0,0]);

%--------------------------------------------------------------------------

function pstruct = makeStruct(parms)
% Convert a cell array of string scalars or char arrays to a parms
% structure.
coder.internal.assert(iscellstr(parms) || isstring(parms), ...
    'MATLAB:mustBeFieldName');
for k = 1:length(parms)
    pstruct.(parms{k}) = uint32(0);
end

%--------------------------------------------------------------------------
