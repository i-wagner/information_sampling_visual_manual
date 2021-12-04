function d=pdistq(x,first,last)
%PDISTQ  Pairwise Euclidean distance
%   D = PDISTQ(X) returns a vector D containing the Euclidean distances between
%   each pair of observations in the M-by-N data matrix X. The M rows of X
%   correspond to observations and the N columns correspond to variables. The
%   output D is a 1-by-M*(M-1)/2 row vector, corresponding to the M*(M-1)/2
%   pairs of observations in X. X may be single or double precision, but must be
%   real and non-sparse.
%
%   D2 = PDISTQ(X, [FIRST LAST]) and D2 = PDISTQ(X, FIRST, LAST) permit subsets
%   of the complete 1-by-M*(M-1)/2 distance vector, D, to be obtained
%   efficiently. D2 is a 1-by-(LAST-FIRST+1) row vector. FIRST and LAST must be
%   single or double precision integers where 1 <= FIRST <= LAST <= M*(M-1)/2.
%
%   Note:
%       PDISTQ(X) is equivalent to PDIST(X) within machine epsilon, EPS.
%
%   See also PDIST, SQUAREFORM, PDIST2.

%   Andrew D. Horchler, adh9 @ case . edu, Created 7-21-13
%   Revision: 1.0, 11-21-14


if ~isfloat(x) || ~isreal(x)
    error('pdistq:InvalidX','Input must be a real floating point matrix.');
end

m = size(x,1);
n = m*(m-1)/2;
if nargin > 1
    if nargin == 2
        if numel(first) ~= 2 || ~isfloat(first) || ~isreal(first)
            error('pdistq:InvalidRange',...
                  'Range must be a real two element floating point vector.');
        end
        last = first(2);
        first = first(1);
    else
        if ~isscalar(first) || ~isfloat(first) || ~isreal(first)
            error('pdistq:InvalidFirst',...
                  'First must be a real floating point scalar.');
        end
        if ~isscalar(last) || ~isfloat(last) || ~isreal(last)
            error('pdistq:InvalidLast',...
                  'Last must be a real floating point scalar.');
        end
    end
    
    if first < 1 || first > n || first ~= floor(first) || ~isfinite(first)
        error('pdistq:InvalidFirstIndex',...
             ['First must be a finite integer greater than or equal to one '...
              'and less than or equal to M*(M-1)/2.']);
    end
    if last < first || last > n || last ~= floor(last) || ~isfinite(last)
        error('pdistq:InvalidLastIndex',...
             ['Last must be a finite integer greater than or equal to First '...
              'and less than or equal to M*(M-1)/2.']);
    end
    n = last-first+1;
end

% Working along rows is faster
x = x.';
d = zeros(1,n,class(x));

k = 1;
if nargin == 1
    for i = 1:m-1
        for j = i+1:m
            d(k) = norm(x(:,i)-x(:,j));
            k = k+1;
        end
    end
else
    l = 1;
    for i = 1:m-1
        for j = i+1:m
            if k >= first
                d(l) = norm(x(:,i)-x(:,j));
                l = l+1;
                if l > n
                    return;
                end
            else
                k = k+1;
            end
        end
    end
end

% ~20% faster, but uses more memory
%{
for i = 1:m-1
    v = bsxfun(@minus,x(:,i),x(:,i+1:m));
    for j = 1:m-i
        d(k) = norm(v(:,j));
        k = k+1;
    end
end
%}