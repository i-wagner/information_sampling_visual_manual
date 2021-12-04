function I = eyeLike(ndiag,m,n,egone)
%MATLAB Code Generation Private Method

%   Copyright 2017-2018 The MathWorks, Inc.
%#codegen
coder.internal.prefer_const(m,n);
ndiagInt = coder.internal.indexInt(ndiag);
I = coder.internal.sparse.spallocLike(m,n,ndiag,egone);
I.colidx(1) = ONE;
I.d(:) = 1;
for c = 2:ndiagInt
    I.colidx(c) = c;
end
for c = ndiagInt+1:coder.internal.indexInt(n)+1
    I.colidx(c) = ndiagInt+1;
end
for r = 1:nnzInt(I)
    I.rowidx(r) = r;
end
coder.internal.sparse.sanityCheck(I);

%--------------------------------------------------------------------------
