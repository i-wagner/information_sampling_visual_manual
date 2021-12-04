function s = parenReference2D(this,r,c)
%MATLAB Code Generation Private Method

% 2D indexing for coder.internal.sparse

%   Copyright 2016-2018 The MathWorks, Inc.
%#codegen

validateIndexTypes(r,c);
if issparse(r)
    s = parenReference2D(this,nonzeros(r),c);
elseif issparse(c)
    s = parenReference2D(this,r,nonzeros(c));
elseif ischar(r) && ischar(c)
    % s(:,:) is just a copy
    coder.inline('always');
    s = this;
elseif ischar(r)
    % s(:,J)
    s = parenReference2DColumns(this,c);
elseif ischar(c)
    % s(I,:)
    s = parenReference2DRows(this,r);
else
    s = parenReference2DNumeric(this,r,c);
end

%--------------------------------------------------------------------------

function s = parenReference2DNumeric(this,r,c)
% Handle indexing: s(r,c) for non-logical arbitrary r and c
validateNumericIndex(ONE,this.m,r);
validateNumericIndex(ONE,this.n,c);
sm = coder.internal.indexInt(numel(r));
sn = coder.internal.indexInt(numel(c));
s = parenReference2DNumericImpl(this,r,c,sm,sn);

%--------------------------------------------------------------------------

function s = parenReference2DNumericImpl(this,r,c,sm,sn)
s = coder.internal.sparse();
ub = sm*sn;
ZERONC = coder.internal.ignoreRange(ZERO);
assert(ZERONC <= ub || ub == coder.internal.indexInt(0)); %<HINT>
s.d = zeros(ZERONC,1,'like',this.d);
s.rowidx = zeros(ZERONC,1,'like',this.rowidx);
s.colidx = zeros(coder.internal.ignoreRange(sn+1),1,'like',this.rowidx);
s.colidx(1) = 1;
colNnz = ONE;
k = ONE;
for cidx = 1:sn
    col = getIdx(c,cidx);
    for ridx = 1:sm
        row = getIdx(r,ridx);
        [idx,found] = locBsearch(this.rowidx,row,this.colidx(col),this.colidx(col+1));
        if found
            s.d = [s.d; this.d(idx)];
            s.rowidx = [s.rowidx; ridx];
            s.d(k) = this.d(idx);
            s.rowidx(k) = ridx;
            k = k+1;
            colNnz = colNnz+1;
        end
    end
    s.colidx(cidx+1) = colNnz;
end
snnz = nnzInt(s);
if snnz == 0
    s.rowidx = ones(coder.internal.ignoreRange(1),1, 'like', ONE);
    s.d = zeros(coder.internal.ignoreRange(1),1, 'like',this.d);
end
s.m = sm;
s.n = sn;
s.maxnz = max2(snnz,ONE);
s.matlabCodegenUserReadableName = makeUserReadableName(s);

%--------------------------------------------------------------------------

function s = parenReference2DColumns(this,c)
% Handle indexing: s(:,c) for numeric c
validateNumericIndex(ONE,this.n,c);
sm = this.m;
sn = coder.internal.indexInt(numel(c));
ub = sm*sn;

% Compute output sizes
nd = coder.internal.indexInt(0);
for cidx = 1:sn
    col = coder.internal.indexInt(c(cidx));
    nd = nd + (this.colidx(col+1) - this.colidx(col));
end
assert(nd <= ub || ub == coder.internal.indexInt(0)); %<HINT>
s = coder.internal.sparse.spallocLike(sm, sn, nd, this);
if nd == 0
    % All zero sparse
    return
end
outIdx = ONE;
for cidx = 1:sn
    col = coder.internal.indexInt(c(cidx));
    colstart = this.colidx(col);
    colend = this.colidx(col+1);
    colNnz =  colend - colstart;
    for k = 1:colNnz
        s.d(outIdx) = this.d(colstart+k-1);
        s.rowidx(outIdx) = this.rowidx(colstart+k-1);
        outIdx = outIdx+1;
    end
    s.colidx(cidx+1) = s.colidx(cidx) + colNnz;
end
s.matlabCodegenUserReadableName = makeUserReadableName(s);

%--------------------------------------------------------------------------

function s = parenReference2DRows(this,r)
% Handle indexing: s(r,:) for numeric r
validateNumericIndex(ONE,this.m,r);
sm = coder.internal.indexInt(numel(r));
sn = this.n;
s = parenReference2DNumericImpl(this,r,':',sm,sn);

%--------------------------------------------------------------------------

function validateIndexTypes(r,c)
validateIndexType(r);
validateIndexType(c);

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
