function this = parenAssign2D(this, rhs, r, c, alreadySqueezed)
%MATLAB Code Generation Private Method

% 2D indexing for coder.internal.sparse

%   Copyright 2016-2019 The MathWorks, Inc.
%#codegen
if nargin < 5
    alreadySqueezed = false;
else
    coder.internal.prefer_const(alreadySqueezed);
end
% Handle any cases like s(1:2,3:4) = ones(1,1,2,1,1,1,2);
if ~alreadySqueezed && ~issparse(rhs) && coder.internal.ndims(rhs) > 2
    this = parenAssign2D(this, squeeze(rhs), r, c, true);
    return
end
validateIndexTypes(r,c);
if ischar(r)
    nr = this.m;
else
    nr = coder.internal.indexInt(numel(r));
end
if ischar(c)
    nc = this.n;
else
    nc = coder.internal.indexInt(numel(c));
end
nAssign = nr*nc;
scalarRhs = isConstScalar(rhs);
scalarR = isConstScalar(r);
scalarC = isConstScalar(c);
vectorRhs = coder.internal.isConst(isvector(rhs)) && isvector(rhs);
vectorLhs = coder.internal.isConst(isvector(this)) && isvector(this);
coder.internal.assert(ismatrix(rhs),'MATLAB:subsassigndimmismatch');
coder.internal.assert(scalarRhs || ...
                      ((scalarR || scalarC) && vectorRhs && sameNumel(rhs,nAssign)) || ...
                      (nr == size(rhs,1) && nc == size(rhs,2)) || ...
                      (vectorRhs && vectorLhs && sameNumel(this,nAssign)), ...
                      'MATLAB:subsassigndimmismatch');
coder.internal.errorIf(isreal(this) && ~isreal(rhs), 'Coder:builtins:LhsRhsComplexMismatch');

if issparse(r)
    this = parenAssign2D(this,rhs,nonzeros(r),c);
elseif issparse(c)
    this = parenAssign2D(this,rhs,r,nonzeros(c));
elseif ischar(r) && ischar(c)
    % s(:,:) = x is just a copy
    coder.inline('always');
    this = parenAssignSpan(this,rhs,scalarRhs,nAssign);
elseif ischar(r)
    % s(:,J)
    if vectorRhs && isrow(rhs)
        this = parenAssign2DColumns(this,rhs.',c);
    else
        this = parenAssign2DColumns(this,rhs,c);
    end
elseif ischar(c)
    % s(I,:)
    if vectorRhs && iscolumn(rhs)
        this = parenAssign2DRows(this,rhs.',r);
    else
        this = parenAssign2DRows(this,rhs,r);
    end
else
    this = parenAssign2DNumeric(this,rhs,r,c);
end

%--------------------------------------------------------------------------

function this = parenAssignSpan(this,rhs,scalarRhs,nAssign)
this = parenAssignAllSpan(this,rhs,scalarRhs,nAssign,2);

%--------------------------------------------------------------------------

function this = parenAssign2DNumeric(this,rhs,r,c)
validateNumericIndex(ONE,this.m,r);
validateNumericIndex(ONE,this.n,c);
sm = coder.internal.indexInt(numel(r));
sn = coder.internal.indexInt(numel(c));
this = parenAssign2DNumericImpl(this,rhs,r,c,sm,sn);

%--------------------------------------------------------------------------

function this = parenAssign2DNumericImpl(this,rhs,r,c,sm,sn)
rhsIter = makeRhsIter(rhs);
for cidx = 1:sn
    col = getIdx(c,cidx);
    for ridx = 1:sm
        row = getIdx(r,ridx);
        [vidx,found] = locBsearch(this.rowidx,row,this.colidx(col),this.colidx(col+1));
        if found
            thisv = this.d(vidx);
        else
            thisv = zeros('like',this.d);
        end
        [rhsv,rhsIter] = nextRhsFromVector(rhs,rhsIter);
        if thisv == 0 && rhsv == 0
            continue
        end
        nz = coder.internal.indexInt(nnz(this));
        if thisv ~= 0 && rhsv ~= 0
            % Just assign value
            this.d(vidx) = rhsv;
        elseif thisv == 0
            % May need to allocate space and shuffle data
            idx = vidx+1;
            if nz == this.maxnz
                % Need to reallocate
                this = realloc(this,nz+10,idx-1,idx,nz,ONE);
                this.rowidx(idx) = row;
                this.d(idx) = rhsv;
            else
                % Just shuffle data right and assign new value
                this = shiftRowidxAndData(this,idx+1,idx,nz-idx+1);
                this.d(idx) = rhsv;
                this.rowidx(idx) = row;
            end
            this = incrColIdx(this,col,ONE);
        else % if rhsv == 0
             % We have memory and are just zeroing out an element. Shift left and decrement.
            idx = vidx;
            this = shiftRowidxAndData(this,idx,idx+1,nz-idx);
            this = decrColIdx(this,col,ONE);
        end
    end
end

%--------------------------------------------------------------------------

function this = parenAssign2DColumns(this,rhs,c)
validateNumericIndex(ONE,this.n,c);
sm = this.m;
sn = coder.internal.indexInt(numel(c));
scalarRhs = isConstScalar(rhs);
rhsIter = makeRhsIter(rhs);
for cidx = 1:sn
    col = coder.internal.indexInt(c(cidx));
    nz = coder.internal.indexInt(nnz(this));
    nzColAlloc = this.colidx(col+1) - this.colidx(col);
    idx = this.colidx(col);
    if scalarRhs
        % Just fill in column with value
        [rhsv,rhsIter] = nextRhs(rhs,rhsIter);
        if rhsv == 0
            % Shift left and decrement
            this = shiftRowidxAndData(this,idx,idx+nzColAlloc,nz-nzColAlloc-idx+1);
            this = decrColIdx(this,col,nzColAlloc);
        else
            % Is there space in this column?
            extraCol = this.m - nzColAlloc;
            if extraCol > 0
                % Check to see if we need to allocate more space
                numAlloc = this.maxnz;
                extraAlloc = numAlloc - nz;
                start = this.colidx(col+1);
                if extraAlloc < extraCol
                    num2Alloc = extraCol-extraAlloc;
                    this = realloc(this,numAlloc+num2Alloc,idx-1,start,nz,extraCol);
                else
                    this = shiftRowidxAndData(this,start+extraCol,start,nz-start+1);
                end
                [this,~,rhsIter] = copyNonzeroValues(this,rhsIter,idx,rhs,rhsv);
                this = incrColIdx(this,col,extraCol);
            else
                % Entire column is nonzero so just assign
                for k = idx:this.colidx(col+1)-1
                    this.d(k) = rhsv;
                end
            end
        end
    else
        % Count non-zeros in corresponding chunk of RHS
        nzRhs = countNumnzInColumn(rhs,rhsIter,sm);

        % Is there space in this column?
        if nzColAlloc < nzRhs
            extraCol = nzRhs - nzColAlloc;
            numAlloc = coder.internal.indexInt(nzmax(this));
            extraAlloc =  numAlloc - nz;
            start = this.colidx(col+1);
            if extraAlloc < extraCol
                num2Alloc =  extraCol - extraAlloc;
                this = realloc(this,numAlloc+num2Alloc,idx-1,start,nz,extraCol);
            else
                this = shiftRowidxAndData(this,start+extraCol,start,nz-start+1);
            end
            [this,~,rhsIter] = copyNonzeroValues(this,rhsIter,idx,rhs);
            this = incrColIdx(this,col,extraCol);
        else
            % Sufficient space
            [this,outIdx,rhsIter] = copyNonzeroValues(this,rhsIter,idx,rhs);

            % Shift data and row indices left and adjust column indices if needed
            extraSpace = nzColAlloc - nzRhs;
            if extraSpace > 0
                start = this.colidx(col+1);
                this = shiftRowidxAndData(this,outIdx,start,nz-start+1);
                this = decrColIdx(this,col,extraSpace);
            end
        end
    end
    rhsIter = nextCol(rhsIter);
end

%--------------------------------------------------------------------------

function this = parenAssign2DRows(this,rhs,r)
validateNumericIndex(ONE,this.m,r);
sm = coder.internal.indexInt(numel(r));
sn = this.n;
this = parenAssign2DNumericImpl(this,rhs,r,':',sm,sn);

%--------------------------------------------------------------------------

function validateIndexTypes(r,c)
validateIndexType(r);
validateIndexType(c);

%--------------------------------------------------------------------------

function y = ONE
y = coder.internal.indexInt(1);

%--------------------------------------------------------------------------

function n = getIdx(idx,k)
% Helper to get index values from either span index, ':', or index vector
coder.inline('always');
if ischar(idx)
    nt = k;
else
    nt = idx(k);
end
n = coder.internal.indexInt(nt);

%--------------------------------------------------------------------------

function this = shiftRowidxAndData(this,outstart,instart,nelem)
if nelem <= zeros('like',nelem)
    return
end
USEMEMMOVE = ~coder.target('MATLAB') && coder.internal.isMemcpyEnabled() && ~coder.internal.isAmbiguousTypes;
if USEMEMMOVE
    this.rowidx = locMemmove(this.rowidx,outstart,instart,nelem);
    this.d = locMemmove(this.d,outstart,instart,nelem);
else
    if outstart >= instart
        for k = nelem-1:-1:0
            this.rowidx(k+outstart) = this.rowidx(k+instart);
            this.d(k+outstart) = this.d(k+instart);
        end
    else
        for k = 0:nelem-1
            this.rowidx(k+outstart) = this.rowidx(k+instart);
            this.d(k+outstart) = this.d(k+instart);
        end
    end
end

%--------------------------------------------------------------------------

function this = incrColIdx(this,col,offs)
for k = col+1:this.n+1
    this.colidx(k) = this.colidx(k)+offs;
end

%--------------------------------------------------------------------------

function this = decrColIdx(this,col,offs)
for k = col+1:this.n+1
    this.colidx(k) = this.colidx(k)-offs;
end

%--------------------------------------------------------------------------

function [this,outIdx,rhsIter] = copyNonzeroValues(this,rhsIter,outStart,rhs,rhsv)
outIdx = outStart;
if nargin == 5
    if rhsv ~= 0
        for k = 1:this.m
            this.rowidx(outIdx) = k;
            this.d(outIdx) = rhsv;
            outIdx = outIdx+1;
        end
    end
elseif issparse(rhs)
    [~,rhsCol] = currentRowCol(rhsIter);
    thisRow = ONE;
    prevRhsRow = ONE;
    for rhsIdx = rhs.colidx(rhsCol):rhs.colidx(rhsCol+1)-1
        rhsRow = rhs.rowidx(rhsIdx);
        thisRow = thisRow+(rhsRow-prevRhsRow);
        this.d(outIdx) = rhs.d(rhsIdx);
        this.rowidx(outIdx) = thisRow;
        outIdx = outIdx+1;
        prevRhsRow = rhsRow;
    end
else
    for k = 1:this.m
        [rhsv,rhsIter] = nextRhs(rhs,rhsIter);
        if rhsv ~= 0
            this.rowidx(outIdx) = k;
            this.d(outIdx) = rhsv;
            outIdx = outIdx+1;
        end
    end
end

%--------------------------------------------------------------------------

function y = locMemmove(y,outstart,instart,nelem)
% Call memmove
voidt = coder.opaque('void');
ignr = coder.internal.opaquePtr('void');
% memmove header will be included by CRL, so don't put include at this point.
ignr = coder.ceval('-jit','memmove', ...
                   coder.ref(y(outstart),'like',voidt), ...
                   coder.rref(y(instart),'like',voidt), ...
                   coder.internal.csizeof(coder.internal.scalarEg(y),nelem)); %#ok<NASGU>

%--------------------------------------------------------------------------

function this = realloc(this,numAllocRequested,ub1,lb2,ub2,offs)
rowidxt = this.rowidx;
dt = this.d;
[~,overflow] = coder.internal.bigProduct(this.m, this.n, true);
if overflow == 0
    numAlloc = max2(ONE,min2(numAllocRequested,coder.internal.indexInt(numel(this))));
else
    numAlloc = max2(ONE,min2(numAllocRequested,intmax(coder.internal.indexIntClass)));
end
this.rowidx = zeros(numAlloc,1,'like',this.rowidx);
this.d = zeros(numAlloc,1,'like',this.d);
this.maxnz = numAlloc;
this.matlabCodegenUserReadableName = makeUserReadableName(this);
for k = 1:ub1
    this.rowidx(k) = rowidxt(k);
    this.d(k) = dt(k);
end

for k = lb2:ub2
    this.rowidx(k+offs) = rowidxt(k);
    this.d(k+offs) = dt(k);
end

%--------------------------------------------------------------------------

function iter = makeRhsIter(rhs)
iter.idx = ONE;
iter.col = ONE;
iter.row = ONE;
scalarRhs = isConstScalar(rhs);
if scalarRhs
    iter.advance = @addNone;
else
    iter.advance = @addOne;
end

%--------------------------------------------------------------------------

function p = isConstScalar(x)
% Is x a fixed-size scalar not including ':'?
coder.inline('always');
p = ~ischar(x) && coder.internal.isConst(isscalar(x)) && isscalar(x);

%--------------------------------------------------------------------------

function [r,c] = currentRowCol(iter)
r = iter.row;
c = iter.col;

%--------------------------------------------------------------------------

function [v,iter] = nextRhs(rhs,iter)
scalarRhs = isConstScalar(rhs);
if scalarRhs
    if issparse(rhs)
        v = rhs.d(1);
    else
        v = rhs(1);
    end
elseif issparse(rhs)

    if iter.idx < rhs.colidx(iter.col+1) && ...
            iter.idx <= nnzInt(rhs) && ...
            iter.row == rhs.rowidx(iter.idx)
        v = rhs.d(iter.idx);
        iter.idx = iter.advance(iter.idx);
    else
        v = zeros('like',rhs.d);
    end
    iter.row = iter.advance(iter.row);
else
    v = rhs(iter.idx);
    iter.idx = iter.advance(iter.idx);
    iter.row = iter.advance(iter.row);
end


%--------------------------------------------------------------------------

function[v, iter] = nextRhsFromVector(rhs, iter)
[v, iter] = nextRhs(rhs, iter);
if issparse(rhs) && iter.row > size(rhs,1)
    iter = nextCol(iter);
end


%--------------------------------------------------------------------------

function iter = nextCol(iter)
iter.col = iter.advance(iter.col);
iter.row = ONE;

%--------------------------------------------------------------------------

function y = addOne(y)
coder.inline('always');
y = y+1;

%--------------------------------------------------------------------------

function y = addNone(y)
coder.inline('always');

%--------------------------------------------------------------------------

function nz = countNumnzInColumn(rhs,rhsIter,sm)
if issparse(rhs)
    [~,col] = currentRowCol(rhsIter);
    nz = rhs.colidx(col+1)-rhs.colidx(col);
else
    nz = zeros('like',ONE);
    for k = 1:sm
        [rhsv,rhsIter] = nextRhs(rhs,rhsIter);
        if rhsv == 0
        else
            nz = nz+1;
        end
    end
end

%--------------------------------------------------------------------------

function y = min2(a,b)
if a <= b
    y = a;
else
    y = b;
end

%--------------------------------------------------------------------------

function p = sameNumel(a,n)
sa = coder.internal.indexInt(size(a));
[lowOrder,highOrder] = coder.internal.bigProduct(sa(1),sa(2),false);
if highOrder == 0
    p = lowOrder == n;
else
    p = false;
end
%--------------------------------------------------------------------------
