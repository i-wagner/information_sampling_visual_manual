function s = binOp(a,b,opstr,sparseOutputPredicate,opImpl)
%MATLAB Code Generation Private Method

% Apply the binary operator specified by opstr or opImpl. To supply an operation by name just supply
% opstr. To override the user-visible name supply opstr as the name to be printed for diagnostics
% and a function handle for opImpl.

% sparseOutputPredicate is a binary function which should determine whether or not the output of
% the binary operation should be sparse. When sparseOutputPredicate is omitted the output is assumed
% to always be sparse.

%   Copyright 2017-2018 The MathWorks, Inc.
%#codegen
coder.internal.prefer_const(opstr);
if nargin < 4
    sparseOutputPredicate = @returnTrue;
end
if nargin < 5
    op = str2func(opstr);
    opImpl = op;
else
    coder.internal.prefer_const(opImpl);
    op = opImpl;
end
opData = getOpData(opstr);
scalara = coder.internal.isConst(isscalar(a)) && isscalar(a);
sparsea = issparse(a);
scalarb = coder.internal.isConst(isscalar(b)) && isscalar(b);
sparseb = issparse(b);
eqsize = ~(scalara || scalarb);
coder.internal.errorIf(isinteger(a) || isinteger(b), ...
                       'MATLAB:sparseInteger');
coder.internal.assert(isAllowedSparseClass(a), ...
                      'Coder:toolbox:unsupportedClass',opstr,class(a));
coder.internal.assert(isAllowedSparseClass(b), ...
                      'Coder:toolbox:unsupportedClass',opstr,class(b));
% Try our hardest to give a compile time error if possible
if eqsize
    if sparsea
        am = a.m;
        an = a.n;
    else
        am = coder.internal.indexInt(size(a,1));
        an = coder.internal.indexInt(size(a,2));
    end
    if sparseb
        bm = b.m;
        bn = b.n;
    else
        bm = coder.internal.indexInt(size(b,1));
        bn = coder.internal.indexInt(size(b,2));
    end
    coder.internal.assert(am == bm, ...
                          'MATLAB:dimagree');
    coder.internal.assert(an == bn, ...
                          'MATLAB:dimagree');
end
if ischar(a)
    s = binOp(double(a),b,opstr,sparseOutputPredicate,opImpl);
    return
end
if ischar(b)
    s = binOp(a,double(b),opstr,sparseOutputPredicate,opImpl);
    return
end
[sm,sn] = getBinOpSize(a,b);
%determine output type and sparsity
ZEROA = fullZero(a);
ZEROB = fullZero(b);
if opData.isDivide || opData.isRem
    isIdent = false;
    ZEROD = coder.internal.scalarEg(ZEROA,ZEROB);
else
    ZEROD = zeros('like',op(ZEROA,ZEROB));
    isIdent = op(ZEROA, ZEROB) == ZEROD;
end

if coder.const(sparseOutputPredicate(a,b))
    coder.internal.assert(ismatrix(a) && ismatrix(b), ...
                          'MATLAB:sparseBinaryOperator:reshapedNdOutput');
end

% Top-level cases
if eqsize
    
    %allocate output
    if isIdent
        sparses = sparseOutputPredicate(a,b);
        temporarilyFull = false;
    else
        sparses = false;
        temporarilyFull = sparseOutputPredicate(a,b);
    end
    ZEROD = op(ZEROA, ZEROB);
    S = allocEqsizeBinop(sparses, opData, ZEROD,a,b,sn,sm);
    
    %fill output
    if sparsea && sparseb
        S = sparseSparseEqsizeBinOp(op,a,b,S);
    else
        if sparsea
            sparseInput = a;
            fullInput = b;
            normalizedOp = op;
        else
            sparseInput = b;
            fullInput = a;
            normalizedOp = @(x,y)(op(y,x));
        end
        S = sparseFullEqsizeBinOp(normalizedOp,sparseInput,fullInput,S);
    end
    
    %ensure output type is correct
    if temporarilyFull
        s = sparse(S);
    else
        s = S;
    end
    
else %scalar expansion
    
    if scalara
        sa = getScalar(a);
        uniOp = @(x)(op(sa,x));
        replaceZerosWith = uniOp(ZEROB);
        c = b;
    else % scalar b
        sb = getScalar(b);
        uniOp = @(x)(op(x,sb));
        replaceZerosWith = uniOp(ZEROA);
        c = a;
    end
    isIdent = replaceZerosWith == ZEROD;

    if ~issparse(c)
        S = uniOp(c);
        if coder.const(sparseOutputPredicate(a,b))
            s = sparse(S);
        else
            s = S;
        end
    elseif isIdent && coder.const(sparseOutputPredicate(a,b))
        s = spfun(uniOp, c);
        return;
    else
        S = eml_expand(replaceZerosWith, sm,sn);
        S = scalarBinOp(uniOp,c,S);
        if coder.const(sparseOutputPredicate(a,b))
            s = sparse(S);
        else
            s = S;
        end
    end
end

%--------------------------------------------------------------------------

function s = sparseSparseEqsizeBinOp(op,a,b,s)
% sparse-sparse same size
coder.internal.prefer_const(op);
ZEROA = fullZero(a);
ZEROB = fullZero(b);
didx = ONE;
if issparse(s)
    s.colidx(ONE) = ONE;
end
n = coder.internal.indexInt(size(s,2));
for c = 1:n
    aidx = a.colidx(c);
    bidx = b.colidx(c);   
    moreAToDo = aidx < a.colidx(c+1);
    moreBToDo = bidx < b.colidx(c+1);
    while (moreAToDo || moreBToDo) %still nonzeros in this col to process
        while (aidx < a.colidx(c+1) &&...
                (~moreBToDo || a.rowidx(aidx) < b.rowidx(bidx)))% process nonzeros in a for rows where b is 0, until we find a nonzero in b or a zero in a
            row = a.rowidx(aidx);
            val = op(a.d(aidx) , ZEROB);
            [s, didx] = writeOneOutput(s, didx, row, c, val);
            aidx = aidx + 1;
        end
        moreAToDo = aidx < a.colidx(c+1);
        while (bidx < b.colidx(c+1) &&...
                (~moreAToDo || b.rowidx(bidx) < a.rowidx(aidx))) %same but a <-> b
            row = b.rowidx(bidx);
            val = op(ZEROA, b.d(bidx));
            [s, didx] = writeOneOutput(s, didx, row, c, val);
            bidx = bidx + 1;
        end
        while (aidx < a.colidx(c+1) && bidx < b.colidx(c+1) && a.rowidx(aidx) == b.rowidx(bidx))%process section where both are non-zero
            row = b.rowidx(bidx);
            val = op(a.d(aidx), b.d(bidx));
            [s, didx] = writeOneOutput(s, didx, row, c, val);
            bidx = bidx + 1;
            aidx = aidx +1;
        end
        moreAToDo = aidx < a.colidx(c+1);
        moreBToDo = bidx < b.colidx(c+1);
    end
    if issparse(s)
        s.colidx(c+1) = didx;
    end
end

%--------------------------------------------------------------------------

function s = sparseFullEqsizeBinOp(op,sparseInput,fullInput,s)
coder.internal.prefer_const(op);
ZEROI = fullZero(sparseInput);

if issparse(s)
    s.colidx(ONE) = ONE;
end

didx = ONE;
sm = coder.internal.indexInt(size(s,1));
sn = coder.internal.indexInt(size(s,2));
for col = 1:sn 
    idx = sparseInput.colidx(col);
    for row = 1:sm
        if idx < sparseInput.colidx(col+1) && row == sparseInput.rowidx(idx)%non-zero element in sparse matrix
            val = op(sparseInput.d(idx), fullInput(row, col));
            idx = idx+1; %look at the next one
        else %zero element
            val = op(ZEROI, fullInput(row, col));
        end
        [s, didx] = writeOneOutput(s, didx, row, col, val);
    end
    if issparse(s)
        s.colidx(col+1) = didx;
    end
end

%--------------------------------------------------------------------------

function s = scalarBinOp(op,c,s)
coder.internal.prefer_const(op);
n = coder.internal.indexInt(size(s, 2));
for col = 1:n
    for idx = c.colidx(col):(c.colidx(col+1)-1)
        row = c.rowidx(idx);
        s(row, col) = op(c.d(idx));
    end
end

%--------------------------------------------------------------------------

function z = fullZero(x)
if issparse(x)
    z = zeros('like',x.d);
else
    z = zeros('like',x);
end

%--------------------------------------------------------------------------

function scalarx = getScalar(x)
sparsex = issparse(x);
if sparsex && coder.internal.indexInt(nnz(x)) > 0
    scalarx = x.d(1);
elseif sparsex
    scalarx = zeros('like',x.d);
else
    scalarx = x(1);
end

%--------------------------------------------------------------------------

function s = allocEqsizeBinop(sparses, opData,ZEROD,a,b,sn,sm)
% Allocate sparse output for op(a,b) where a and b have equal sizes

% prefer_const to preserve constant sizes in allocation
coder.internal.prefer_const(sparses, opData,ZEROD,sn,sm);

if ~sparses %includes non-identity case
    s = eml_expand(ZEROD,sm,sn);
   return; 
end

nza = coder.internal.indexInt(nnz(a));
nzb = coder.internal.indexInt(nnz(b));

if opData.isAnd
    numalloc = min(nza,nzb);
else
    coder.internal.assert( ...
        nza <= intmax(coder.internal.indexIntClass) - nzb ||... %check that add is safe
        mulIsSafe(sn,sm),...
        'Coder:toolbox:SparseFuncAlmostFull');
    numalloc = min(nza + nzb, sn*sm);
end
numalloc = max2(numalloc,ONE);
s = coder.internal.sparse.spallocLike(sm, sn, numalloc, ZEROD);


%--------------------------------------------------------------------------

function [s,didx] = writeOneOutput(s, didx, row, col, val)

if issparse(s)
    if val ~= zeros(1,1, 'like', val) %otherwise, nothing to do
        s.d(didx) = val;
        s.rowidx(didx) = row;
        didx = didx+1;
    end
else
    s(row, col) = val;
end



%--------------------------------------------------------------------------

function [m,n] = getBinOpSize(a,b)
f = @coder.internal.indexInt;
if coder.internal.isConst(isscalar(a)) && isscalar(a)
    m = f(size(b,1));
    n = f(size(b,2));
elseif coder.internal.isConst(isscalar(b)) && isscalar(b)
    m = f(size(a,1));
    n = f(size(a,2));
else
    if coder.internal.isConst(size(a,1))
        m = f(size(a,1));
    else
        m = f(size(b,1));
    end
    if coder.internal.isConst(size(a,2))
        n = f(size(a,2));
    else
        n = f(size(b,2));
    end
end

%--------------------------------------------------------------------------

function p = returnTrue(~,~)
coder.inline('always');
p = true;

%--------------------------------------------------------------------------

function p = getOpData(op)
coder.internal.prefer_const(op);
p.isAnd = coder.const(strcmp(op,'and'));
p.isDivide = coder.const(strcmp(op,'rdivide') || strcmp(op,'ldivide'));
p.isRem = coder.const(strcmp(op, 'rem'));

%--------------------------------------------------------------------------

function out = mulIsSafe(a,b)
[~,overflow] = coder.internal.bigProduct(a,b,true);
out = overflow==0;
