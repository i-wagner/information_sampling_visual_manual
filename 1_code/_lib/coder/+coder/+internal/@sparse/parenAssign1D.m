function this = parenAssign1D(this, rhs, linidx)
%MATLAB Code Generation Private Method

% 1D indexing for coder.internal.sparse

%   Copyright 2016-2018 The MathWorks, Inc.
%#codegen
validateIndexType(linidx);
[lowOrderSize, highOrderSize] = coder.internal.bigProduct(this.m,this.n,false);
scalarRhs = coder.internal.isConst(isscalar(rhs)) && isscalar(rhs);
coder.internal.assert(goodNAssign(lowOrderSize,highOrderSize,rhs,linidx) || scalarRhs, ...
                      'MATLAB:subsassignnumelmismatch');
coder.internal.errorIf(isreal(this) && ~isreal(rhs), 'Coder:builtins:LhsRhsComplexMismatch');

if issparse(linidx)
    this = parenAssign1D(this,rhs,nonzeros(linidx));
elseif ischar(linidx)
    % s(:) = x is just a copy
    coder.inline('always');
    this = parenAssign1DSpan(this,rhs,scalarRhs, highOrderSize);
else
    this = parenAssign1DNumeric(this,rhs,linidx, highOrderSize);
end

%--------------------------------------------------------------------------

function this = parenAssign1DSpan(this,rhs,scalarRhs,overflow)
coder.inline('always');
coder.internal.assert(overflow == 0, 'Coder:toolbox:SparseColonOverflow');
this = parenAssignAllSpan(this,rhs,scalarRhs,this.m*this.n,1);

%--------------------------------------------------------------------------

function this = parenAssign1DNumeric(this,rhs,linidx, overflow)
if overflow == 0
    validateNumericIndex(ONE,this.m*this.n,linidx);
else
    validateNumericIndex(ONE,intmax(coder.internal.indexIntClass),linidx);
end
scalarRhs = coder.internal.isConst(isscalar(rhs)) && isscalar(rhs);
nidx = coder.internal.indexInt(numel(linidx));
for k = 1:nidx
    idx = linidx(k);
    [row,col] = ind2sub(size(this),idx);
    if scalarRhs
        rhsv = full(rhs(1,1));
    else
        rhsv = full(rhs(k));
    end
    this = parenAssign2D(this,rhsv,row,col);
end

%--------------------------------------------------------------------------

function out = goodNAssign(sizeLOW,sizeHIGH,rhs,linidx)
if issparse(rhs)
    sr = coder.internal.indexInt(size(rhs));
    [lowOrderRHS, highOrderRHS] = coder.internal.bigProduct(sr(1),sr(2),false); 
    if ischar(linidx)
        lowOrderLHS = sizeLOW;
        highOrderLHS = sizeHIGH;
    else
        sl = coder.internal.indexInt(size(linidx));
        [lowOrderLHS, highOrderLHS] = coder.internal.bigProduct(sl(1),sl(2),false);
    end
    out = lowOrderLHS==lowOrderRHS && highOrderLHS == highOrderRHS;
else
    if ischar(linidx)
        out = numel(rhs) == sizeLOW;
    else
        out = numel(rhs) == numel(linidx);
    end
end


%--------------------------------------------------------------------------

function y = ONE
y = coder.internal.indexInt(1);

%--------------------------------------------------------------------------
