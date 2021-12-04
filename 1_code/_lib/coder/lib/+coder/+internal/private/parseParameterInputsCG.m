function pstruct = parseParameterInputsCG(parms,options,varargin)
%MATLAB Code Generation Private Function

%   Version of parseParameterInputs to be executed by the codegen
%   constant-folder.

%   Copyright 2009-2019 The MathWorks, Inc.
%#codegen

coder.inline('always');
coder.internal.allowEnumInputs;
coder.internal.allowHalfInputs;
narginchk(2,inf);
coder.internal.prefer_const(parms,options);
if isstruct(parms)
    pstruct = parseParameterInputsCG( ...
        coder.const(fieldnames(parms)),options,varargin{:});
    return
end
coder.internal.assert(iscell(parms), ...
    'Coder:toolbox:eml_parse_parameter_inputs_2', ...
    'IfNotConst','Fail');
[caseSensitive,partialMatch,expandStructs,ignoreNulls] = ...
    coder.const(@processOptions,options);
[isMatch,isExactMatch] = coder.const(@parmMatchFunction, ...
    caseSensitive,partialMatch);
nargs = nargin - 2;
% These are technical limitations of this implementation, so we check them
% here, regardless of whether another limitation may make them impossible
% to violate.
coder.internal.assert(nargs <= 65535, ...
    'Coder:toolbox:eml_parse_parameter_inputs_3', ...
    'IfNotConst','Fail');
nparms = coder.const(length(parms));
coder.internal.assert(nparms <= 65535, ...
    'Coder:toolbox:eml_parse_parameter_inputs_4', ...
    'IfNotConst','Fail');
% Create and initialize the output structure.
pstruct = coder.const(makeStruct(parms));
if nargs > 0
    % Parse VARARGIN{:}.
    t = coder.const(inputTypes(varargin{:}));
    coder.internal.assert(t(nargs) ~= 'n', ...
        'Coder:toolbox:eml_parse_parameter_inputs_6',varargin{nargs}, ...
        'IfNotConst','Fail');
    coder.unroll;
    for k = 1:nargs
        if coder.const(t(k) == 'n') % name
            % Find the index of the field varargin{k} in PARMS.
            pidx = coder.const(findParm(varargin{k},parms, ...
                isMatch,isExactMatch,caseSensitive,partialMatch));
            % The parameter value is in varargin{k+1}. Set the value of
            % the field in PARMS accordingly.
            if coder.const(~ignoreNulls || ~isnull(varargin{k+1}))
                pstruct.(parms{pidx}) = coder.const(uint32(k+1));
            end
        elseif coder.const(expandStructs && coder.const(t(k) == 's')) % struct
            coder.unroll;
            for fieldidx = 0:eml_numfields(varargin{k})-1
                fname = eml_getfieldname(varargin{k},fieldidx);
                if coder.const(~ignoreNulls || ~isnull(varargin{k}.(fname)))
                    % Find the index of the corresponding field in PARMS.
                    pidx = coder.const(findParm(fname,parms,...
                        isMatch,isExactMatch,caseSensitive,partialMatch));
                    % The parameter value is in the struct varargin{k} at
                    % field index fieldidx. Set the value of the field
                    % in PARMS accordingly.
                    pstruct.(parms{pidx}) = coder.const( ...
                        combineIndices(uint32(k),uint32(fieldidx)));
                end
            end
        else
            % Last entry must be a value if it is not a structure.
            coder.internal.assert(t(k) == 'v', ...
                'Coder:toolbox:eml_parse_parameter_inputs_8', ...
                'IfNotConst','Fail');
        end
    end
end

%--------------------------------------------------------------------------

function [caseSensitive,partialMatch,expandStructs,ignoreNulls] = ...
    processOptions(options)
% Extract parse options from options input structure, supplying default
% values if needed.
coder.internal.allowEnumInputs;
coder.internal.prefer_const(options);
coder.internal.assert(coder.internal.isConst(options), ...
    'Coder:toolbox:eml_parse_parameter_inputs_9', ...
    'IfNotConst','Fail');
% Set defaults.
caseSensitive = false;
expandStructs = true;
partialMatch = 'n'; % none
ignoreNulls = false;
% Read options.
if ~isempty(options)
    coder.internal.assert(isstruct(options), ...
        'Coder:toolbox:eml_parse_parameter_inputs_10', ...
        'IfNotConst','Fail');
    fnames = coder.const(fieldnames(options));
    nfields = coder.const(length(fnames));
    coder.unroll;
    for k = 1:nfields
        fname = fnames{k};
        if coder.const(strcmp(fname,'CaseSensitivity'))
            coder.internal.assert(isscalar(options.CaseSensitivity) && ...
                islogical(options.CaseSensitivity), ...
                'Coder:toolbox:eml_parse_parameter_inputs_11', ...
                'IfNotConst','Fail');
            caseSensitive = coder.const(options.CaseSensitivity);
        elseif coder.const(strcmp(fname,'StructExpand'))
            coder.internal.assert(isscalar(options.StructExpand) && ...
                islogical(options.StructExpand), ...
                'Coder:toolbox:eml_parse_parameter_inputs_12', ...
                'IfNotConst','Fail');
            expandStructs = coder.const(options.StructExpand);
        elseif coder.const(strcmp(fname,'PartialMatching'))
            isfirst = strcmp(options.PartialMatching,'first') || ( ...
                isscalar(options.PartialMatching) && ...
                options.PartialMatching ~= false);
            isnone = strcmp(options.PartialMatching,'none') || ( ...
                isscalar(options.PartialMatching) && ...
                options.PartialMatching == false);
            isunique = strcmp(options.PartialMatching,'unique');
            coder.internal.assert(isfirst || isnone || isunique, ...
                'Coder:toolbox:eml_parse_parameter_inputs_13', ...
                'IfNotConst','Fail');
            if isunique
                partialMatch = 'u'; % unique
            elseif isfirst
                partialMatch = 'f'; % first
            else
                partialMatch = 'n'; % none
            end
        elseif coder.const(strcmp(fname,'IgnoreNulls'))
            coder.internal.assert(isscalar(options.IgnoreNulls) && ...
                islogical(options.IgnoreNulls), ...
                'Coder:toolbox:BadIgnoreNulls', ...
                'IfNotConst','Fail');
            ignoreNulls = coder.const(options.IgnoreNulls);
        else
            coder.internal.assert(false, ...
                'Coder:toolbox:eml_parse_parameter_inputs_14', ...
                'IfNotConst','Fail');
        end
    end
end

%--------------------------------------------------------------------------

function t = inputTypes(varargin)
% Returns an array indicating the classification of each argument as a
% parameter name, parameter value, option structure, or unrecognized. The
% return value must be constant folded.
coder.internal.allowEnumInputs;
t = coder.nullcopy(char(zeros(nargin,1)));
isval = false;
coder.unroll;
for k = 1:nargin
    if isval
        t(k) = 'v'; % value
        isval = false;
    elseif coder.internal.isTextRow(varargin{k})
        coder.internal.assert(coder.internal.isConst(varargin{k}), ...
            'Coder:toolbox:eml_parse_parameter_inputs_15', ...
            'IfNotConst','Fail');
        t(k) = 'n'; % name
        isval = true;
    elseif isstruct(varargin{k})
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
coder.inline('always');
coder.internal.prefer_const(parm,parms,matchFun,exactMatchFun, ...
    caseSensitive,partialMatch);
[n,ncandidates] = findParmKernel(parm,parms,matchFun,exactMatchFun, ...
    partialMatch);
coder.internal.assert(ncandidates ~= 0, ...
    'Coder:toolbox:eml_parse_parameter_inputs_16',parm, ...
    'IfNotConst','Fail');
if ncandidates ~= 1
    coder.internal.assert(false, ...
        'Coder:toolbox:AmbiguousPartialMatch',parm, ...
        coder.const(feval('coder.internal.partialParameterMatchString', ...
        coder.internal.toCharIfString(parm),parms,caseSensitive)));
end

%--------------------------------------------------------------------------

function [n,ncandidates] = findParmKernel(parm,parms,isMatch, ...
    isExactMatch,partialMatch)
coder.inline('always');
coder.internal.prefer_const(parm,parms,isMatch,isExactMatch,partialMatch);
uPartMatch = partialMatch == 'u'; % unique partial matching
n = 0;
ncandidates = 0;
nparms = coder.const(length(parms));
for j = 1:nparms
    if coder.const(isMatch(parms{j},parm))
        if coder.const(isExactMatch(parms{j},parm))
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

function [isMatch,isExactMatch] = parmMatchFunction(casesens,prtmatch)
% Return function handles for matching of parameter names. The matchFun
% respects the casesens and prtmatch options. The exactMatchFun replaces
% prtmatch with PM_NONE (no partial matching). Note that if prtmatch is
% PM_NONE to begin with, exactMatchFun simply returns true without doing
% any work, since it is meant to be used only after detecting a match with
% matchFun.
coder.internal.prefer_const(casesens,prtmatch);
partial = coder.const(prtmatch ~= 'n');
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
coder.inline('always');
coder.internal.prefer_const(mstrparm,userparm);
if coder.const(isempty(userparm))
    p = false;
else
    p = coder.const(strncmp(mstrparm,userparm,strlength(userparm)));
end

function p = isCaseInsensitivePartialMatch(mstrparm,userparm)
coder.inline('always');
coder.internal.prefer_const(mstrparm,userparm);
if coder.const(isempty(userparm))
    p = false;
else
    p = coder.const(strncmpi(mstrparm,userparm,strlength(userparm)));
end

function p = isCaseSensitiveMatch(mstrparm,userparm)
coder.inline('always');
coder.internal.prefer_const(mstrparm,userparm);
if coder.const(isempty(userparm))
    p = false;
else
    p = coder.const(strcmp(mstrparm,userparm));
end

function p = isCaseInsensitiveMatch(mstrparm,userparm)
coder.inline('always');
coder.internal.prefer_const(mstrparm,userparm);
if coder.const(isempty(userparm))
    p = false;
else
    p = coder.const(strcmpi(mstrparm,userparm));
end

function p = returnTrue(mstrparm,userparm)
coder.inline('always');
coder.internal.prefer_const(mstrparm,userparm);
p = true;

%--------------------------------------------------------------------------

function n = combineIndices(vargidx,stfldidx)
% Returns a 'uint32'. Stores the struct field index (zero-based) in the
% low bits and the varargin index in the low bits.
% n = (struct_field_ordinal << 16) + vargidx;
coder.internal.prefer_const(vargidx,stfldidx);
n = coder.const(eml_bitor(eml_lshift(vargidx,int8(16)),stfldidx));

%--------------------------------------------------------------------------

function p = isnull(x)
% Returns true if x is [] and fixed-size.
coder.inline('always');
p = coder.const(isa(x,'double') && coder.internal.isConst(size(x)) && ...
    isequal(size(x),[0,0]));

%--------------------------------------------------------------------------

function pstruct = makeStruct(parms)
% Convert a cell array of string scalars or char arrays to a parms
% structure.
coder.internal.prefer_const(parms);
coder.internal.assert(coder.internal.isConst(parms), ...
    'Coder:toolbox:InputMustBeConstant','parms');
coder.unroll;
for k = 1:length(parms)
    coder.internal.assert(coder.internal.isTextRow(parms{k}), ...
        'MATLAB:mustBeFieldName');
    pstruct.(parms{k}) = uint32(0);
end

%--------------------------------------------------------------------------
