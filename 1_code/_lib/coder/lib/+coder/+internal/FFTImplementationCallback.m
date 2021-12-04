classdef (Abstract) FFTImplementationCallback
%MATLAB Code Generation Private Class

%   Copyright 2019-2020 The MathWorks, Inc.

%FFTImplementationCallback : Inherit from this class to implement the
%FFTs using Radix-2 or Bluestein algorithm. This class provides all the
%helper functions required to implement the FFT using the above two
%algorithms.

%#codegen
    methods (Access = protected)
        function y = fftReferenceCall(this,x,nfft,isInverse)
            coder.inline('always');
            nfft = cast(nfft,this.ucls);
            algid = this.AUTO;
            useRadix2 = this.get_size_props(algid, nfft);
            % Select twiddle option:
            twidopt = this.FULL_TWIDDLE_TABLE;
            [N2blue, nRows] = this.get_algo_sizes(nfft, useRadix2);
            [costab, sintab, sintabinv] = ...
                this.generate_twiddle_tables(x, isInverse, nRows, twidopt, useRadix2);
            if useRadix2
                y = this.r2br_r2dit_trig(x,nfft,isInverse,costab,sintab,twidopt);
            else
                y = this.dobluesteinfft(x,N2blue,nfft,isInverse,costab,sintab,sintabinv,twidopt);
            end
        end
        function y = dobluesteinfft(this,x,n2blue,nfft,isInverse,costab,sintab,sintabinv,twidopt)
        % Helper to perform computation with Bluestein
            ZERO = (coder.internal.indexInt(0));
            TWO = coder.internal.indexInt(2);
            nChan = coder.internal.prodsize(x,'except',1);

            % Check to see if half length optimization can be applied
            ishalflength = isreal(x) && (nfft ~= 1) && bitand(cast(nfft,this.ucls),ones(this.ucls)) == 0;

            % Generate twiddle table to preprocess inputs before feeding to bluestein
            % If half length optimization can be applied, then generate twiddle table
            % to preprocess N/2 point fft.
            if ishalflength
                wwc = coder.internal.bluesteinSetup(coder.internal.scalarEg(x),coder.internal.indexDivide(nfft,TWO),isInverse);
            else
                wwc = coder.internal.bluesteinSetup(coder.internal.scalarEg(x),nfft,isInverse);
            end

            % If passed compile-time vector skip parfor logic
            if coder.internal.isConst(nChan) && nChan == 1
                y = this.bluestein(x,ZERO,n2blue,nfft,isInverse,costab,sintab,costab,sintabinv,twidopt,wwc);
            else
                nrows = coder.internal.indexInt(size(x,1));
                if coder.internal.useParforConst('fft',n2blue)
                    % Compute transform one channel at a time.
                    oneChanBluestein = true;
                    y = coder.internal.fft.allocFftOutput(x,nfft);
                    parfor chan = 1:nChan
                        xoff = (chan-1)*nrows;
                        y(:,chan) = bluestein(this,x,xoff,n2blue,nfft,isInverse,costab,sintab,costab,sintabinv,twidopt,wwc,oneChanBluestein);
                    end
                else
                    y = this.bluestein(x,ZERO,n2blue,nfft,isInverse,costab,sintab,costab,sintabinv,twidopt,wwc);
                end
            end

        end
        function y = r2br_r2dit_trig(this,x,n1_unsigned,isInverse,costab,sintab,twidopt)
        % Bit-reverse and then do in-place radix-2 decimation-in-time FFT
        % Trig-based twiddle computation.
        % Pad or clip dimension 1 of input x to be unsigned_nRows long.
            coder.internal.prefer_const(n1_unsigned);
            coder.internal.prefer_const(isInverse);

            % Define constants.
            ZERO = coder.internal.indexInt(0);
            nChan = coder.internal.prodsize(x,'except',1);
            n1 = coder.internal.indexInt(n1_unsigned);
            % If passed compile-time vector skip parfor logic
            if coder.internal.isConst(nChan) && nChan == 1
                y = this.r2br_r2dit_trig_impl(x,ZERO,n1_unsigned,isInverse,costab,sintab,twidopt);
            else
                nrows = coder.internal.indexInt(size(x,1));
                if coder.internal.useParforConst('fft',n1)
                    % Preallocate output.
                    y = coder.internal.fft.allocFftOutput(x,n1_unsigned);
                    % Compute transform one channel at a time.
                    oneChanRadix2 = true;
                    parfor chan = 1:nChan
                        xoff = (chan-1)*nrows;
                        y(:,chan) = r2br_r2dit_trig_impl(this,x,xoff,n1_unsigned,isInverse,costab,sintab,twidopt,oneChanRadix2);
                    end
                else
                    y = this.r2br_r2dit_trig_impl(x,ZERO,n1_unsigned,isInverse,costab,sintab,twidopt);
                end
            end
            % Rescaling in the ifft case.
            if isInverse && (size(y,1) > 1)
                r = eml_rdivide(ones('like',real(y)),size(y,1));
                y = y * r;
            end

        end
    end
    methods (Sealed,Access = protected)
        function y = bluestein(this,x,xoffInit,nfft,nRows,isInverse,costab,sintab,costabinv,sintabinv,twidopt,wwc,oneChan)
            coder.inline('always');
            % Define constants
            ONE = coder.internal.indexInt(1);
            nRowsIdx = coder.internal.indexInt(nRows);

            nrowsx = coder.internal.indexInt(size(x,1));

            if nargin >= 13
                coder.internal.prefer_const(oneChan);
            else
                oneChan = false;
            end
            if oneChan
                nChan = ONE;
                % Preallocate output for a single column only
                y = coder.internal.fft.allocFftOutput(zeros(nrowsx,ONE,'like',x),nRows);
            else
                nChan = coder.internal.prodsize(x,'except',1);
                % Preallocate output.
                y = coder.internal.fft.allocFftOutput(x,nRows);
            end

            % Do half length algorithm if nfft is even and x is real valued
            ishalflength = isreal(x) && (nfft ~= 1) && bitand(cast(nRows,this.ucls),ones(this.ucls)) == 0;
            if ishalflength
                y = this.doHalfLengthBluestein(x,xoffInit,y,nChan,nrowsx,nRows,nfft,wwc,isInverse,twidopt, ...
                                               costab,sintab,costabinv,sintabinv,nRowsIdx);
            else
                y = this.doNonHalfLengthBluestein(x,xoffInit,y,nChan,nrowsx,nRows,nfft,wwc,isInverse,twidopt, ...
                                                  costab,sintab,costabinv,sintabinv,nRowsIdx);
            end
        end
        function y = doNonHalfLengthBluestein(this,x,xoffInit,y,nChan,nrowsx,nRows,nfft,wwc,isInverse,twidopt, ...
                                              costab,sintab,costabinv,sintabinv,nRowsIdx)
            coder.inline('always');
            % Define constants
            ONE = coder.internal.indexInt(1);
            minNrowsNx = eml_min(nRowsIdx, coder.internal.indexInt(size(x,1)));
            % Floating point indices for arithmetic
            n = cast(nRows,'like',real(x));
            m = length(wwc);
            % FFT computation using bluestein performed one column at a time
            for chan = 1:nChan
                xoff = (chan-1)*nrowsx+xoffInit;
                yoff = (chan-1)*coder.internal.indexInt(nRows);
                xidx = ONE+xoff;
                % Preprocessing x before performing bluestein
                for k = 1:minNrowsNx
                    y(yoff+k) = coder.internal.conjtimes(wwc(nRowsIdx+k-1),x(xidx));
                    xidx = xidx+1;
                end
                % Initialize remaining elements to prevent unexpected results
                for k = minNrowsNx+1:nRowsIdx
                    y(yoff+k) = 0;
                end
                % Calculate N point FFT using the bluestein algorithm
                y = this.bluesteinAlgo(x,y,yoff,costab,sintab,twidopt,nfft,wwc, ...
                                       costabinv,sintabinv,isInverse,nRows,m,n);
            end
        end
        function y = doHalfLengthBluestein(this,x,xoffInit,y,nChan,nrowsx,nRows,nfft,wwc,isInverse,twidopt, ...
                                           costab,sintab,costabinv,sintabinv,nRowsIdx)
            coder.inline('never');
            % Define required constants
            ONE = coder.internal.indexInt(1);
            TWO = coder.internal.indexInt(2);
            algo = this.BLUESTEIN;
            hnRows = coder.internal.indexDivide(nRows,TWO);
            n1 = coder.internal.indexInt(nRows);

            % Create temporary array for performing half length optimization
            % Preallocate output for a single column only
            ytmp = coder.internal.fft.allocFftOutput(zeros(nrowsx,ONE,'like',x),hnRows);

            % Pad/truncate for half length optimization
            [nxeven,minHnrowsNx] = this.padOrTruncate(x,n1);

            % Floating point indices for arithmetic
            hn = cast(hnRows,'like',real(x));
            hm = length(wwc);

            % sin/cos table generation required to reconstruct N point fft from the
            % N/2 point fft obtained from half length optimization
            [costable, sintable, ~] = this.generate_twiddle_tables(x, isInverse, ...
                                                              coder.internal.indexTimes(nRows,TWO), twidopt, false);
            % Get twiddle factors to compute N/2 point fft
            [hcostab,hsintab,hcostabinv,hsintabinv] = this.get_half_twiddle_tables(x,algo, ...
                                                                                     costab,sintab,costabinv,sintabinv);
            % Get reconstruction factors for reconstructing original fft
            [reconVar1,reconVar2] = this.get_reconstruct_factors(algo,x,hnRows, ...
                                                                      costable,sintable,isInverse);
            % Get wrap around index for reconstructing original fft
            wrapIndex = this.calculate_wrapIndex(hnRows);

            % Half length optimization with bluestein performed one column at a time
            for chan = 1:nChan
                xoff = (chan-1)*nrowsx+xoffInit;
                yoff = (chan-1)*nRowsIdx;

                % Create complex number from the real values of x and perform
                % bluestein preprocessing for N/2 point FFT
                xidx = ONE+xoff;
                for k1 = 1:minHnrowsNx/2
                    ytmp(k1) = coder.internal.conjtimes(wwc(hnRows+k1-1),complex(x(xidx),x(xidx+1)));
                    xidx = xidx+2;
                end

                % When x is odd length the last complex number should be
                % complex(x(idx),0)
                % Initialize remaining elements to prevent unexpected results
                if ~nxeven
                    ytmp((minHnrowsNx/2)+1) = coder.internal.conjtimes(wwc(hnRows+(minHnrowsNx/2)), ...
                                                                      complex(x(xidx)));
                    if ((minHnrowsNx/2)+2 <= hnRows)
                        for i = minHnrowsNx/2+2:hnRows
                            ytmp(i) = 0;
                        end
                    end
                else
                    if minHnrowsNx/2+1 <= hnRows
                        for i = minHnrowsNx/2+1:hnRows
                            ytmp(i) = 0;
                        end
                    end
                end

                % Calculate N/2 point FFT on the N/2 length complex signal
                ytmp = this.bluesteinAlgo(x,ytmp,0,hcostab,hsintab,twidopt,nfft/2,wwc, ...
                                          hcostabinv,hsintabinv,isInverse,hnRows,hm,hn);

                % Get back N point fft from N/2 point fft
                y = this.getback_bluestein_fft(y,ytmp,yoff,0,reconVar1,reconVar2,wrapIndex,hnRows);
            end
            % Scaling for ifft computation
            if isInverse
                y = y/2;
            end
        end
        function y = r2br_r2dit_trig_impl(this,x,xoffInit,unsigned_nRows,isInverse,costab,sintab,twidopt,oneChan)
            coder.internal.prefer_const(unsigned_nRows,isInverse);
            % Define constants.
            ONE = coder.internal.indexInt(1);
            nrowsx = coder.internal.indexInt(size(x,1));

            if nargin >= 9
                coder.internal.prefer_const(oneChan);
            else
                oneChan = false;
            end
            if oneChan
                nChan = ONE;
                % Preallocate output for a single column only
                y = coder.internal.fft.allocFftOutput(zeros(nrowsx,ONE,'like',x),unsigned_nRows);
            else
                nChan = coder.internal.prodsize(x,'except',1);
                % Preallocate output.
                y = coder.internal.fft.allocFftOutput(x,unsigned_nRows);
            end

            % Do half length algorithm if nfft is even and x is real valued
            % Here nfft is always even as nfft is radix 2
            useHalfLength = isreal(x) && (unsigned_nRows ~= 1);
            if useHalfLength
                y = this.doHalfLengthRadix2(x,xoffInit,y,nChan,unsigned_nRows,nrowsx,costab,sintab,twidopt,isInverse);
            else
                y = this.doNonHalfLengthRadix2(x,xoffInit,y,nChan,unsigned_nRows,nrowsx,costab,sintab,twidopt,isInverse);
            end
        end
        function y = doNonHalfLengthRadix2(this,x,xoffInit,y,nChan,unsigned_nRows,nrowsx,costab,sintab,twidopt,isInverse)
            coder.inline('always');
            % Define constants
            ONE = coder.internal.indexInt(1);
            nRows = coder.internal.indexInt(unsigned_nRows);
            [nRowsM1,nRowsM2,nRowsD2,nRowsD4,e] = this.radix2_constants(x,real(x),nRows);

            % FFT computation using radix2 performed one column at a time
            for chan = 1:nChan
                xoff = (chan-1)*nrowsx+xoffInit;
                yoff = (chan-1)*nRows;
                chanStart = yoff;
                % Initialize column of y with bitreversed complex copy of the
                % corresponding column of x.
                ix = xoff;
                iy = yoff;
                ju = zeros(this.ucls);
                for i = ONE:nRowsM1
                    y(iy+1) = x(ix+1);
                    ju = this.fft_bitrevidx(ju,unsigned_nRows);
                    iy = chanStart+eml_cast(ju,coder.internal.indexIntClass,'wrap');
                    ix = ix+1;
                end
                y(iy+1) = x(ix+1);

                % Calculate N point FFT using the radix2 algorithm
                y = this.radix2Algo(y,chanStart,twidopt,costab,sintab,nRows, ...
                                    nRowsM2,nRowsD2,nRowsD4,e,isInverse);
            end
        end
        function y = doHalfLengthRadix2(this,x,xoffInit,y,nChan,unsigned_nRows,nrowsx,costab,sintab,twidopt,isInverse)
            coder.inline('never');
            % Define constants
            ONE = coder.internal.indexInt(1);
            TWO = coder.internal.indexInt(2);
            algo = this.RADIX2;
            n1 = coder.internal.indexInt(unsigned_nRows);
            nRows = coder.internal.indexDivide(n1,TWO);
            [nRowsM1,nRowsM2,nRowsD2,nRowsD4,e] = this.radix2_constants(y,real(x),nRows);

            % Get twiddle factors to compute N/2 point fft
            [hcostab,hsintab,~,~] = this.get_half_twiddle_tables(x,algo,costab,sintab,[],[]);
            % Get reconstruction factors for reconstructing original fft
            [reconVar1,reconVar2] = this.get_reconstruct_factors(algo,x,nRows, ...
                                                                      costab,sintab,isInverse);
            % Get wrap around index for reconstructing original fft
            wrapIndex = this.calculate_wrapIndex(nRows);

            % Bitreversed index calculation for in place fft algorithm
            bitrevIndex = this.get_bitrevIndex(nRowsM1,unsigned_nRows/2);

            % Pad/truncate for half length optimization
            [nxeven,minX] = this.padOrTruncate(x,n1);

            % Half length with radix2 performed one column at a time
            for chan = 1:nChan
                xoff = (chan-1)*nrowsx+xoffInit;
                yoff = (chan-1)*n1;
                chanStart = yoff;
                % Copy the elements of x in bitreversed order to y to form complex pair
                ix = xoff;
                for i = ONE:minX/2
                    y(yoff+bitrevIndex(i)) = complex(x(ix+1),x(ix+2));
                    ix = ix + 2;
                end
                % When x is odd length the last complex number should be complex(x(idx),0)
                if ~nxeven
                    y(yoff+bitrevIndex((minX/2)+1)) = complex(x(ix+1));
                end

                % Calculate N/2 point FFT on the N/2 length complex signal
                y = this.radix2Algo(y,chanStart,twidopt,hcostab,hsintab,nRows, ...
                                    nRowsM2,nRowsD2,nRowsD4,e,isInverse);

                % Get back N point fft from N/2 point fft
                y = this.getback_radix2_fft(y,yoff,reconVar1,reconVar2,wrapIndex,n1/2);
            end
        end
        function y = bluesteinAlgo(this,x,y,yoff,costab,sintab,twidopt,nfft,wwc, ...
                                   costabinv,sintabinv,isInverse,nRows,m,n)
            % Bluestein algorithm to be applied after input preprocessing
            coder.inline('always');
            % Define constants
            ONE = coder.internal.indexInt(1);

            %------- Fast convolution via FFT.
            N2U = cast(nfft,this.ucls);
            % fy = fft(y, nfft);
            % Only use one channel from y
            oneChanRadix2 = true;
            fy = ...
                this.r2br_r2dit_trig_impl(y,yoff,N2U,false,costab,sintab,twidopt,oneChanRadix2);
            % fv = fft(conj(ww), nfft);   % <----- Chirp filter.
            fv = ...
                this.r2br_r2dit_trig(wwc,N2U,false,costab,sintab,twidopt);

            fy = fy .* fv;
            % fy  = ifft(fy);
            fv = ...
                this.r2br_r2dit_trig(fy,N2U,true,costabinv,sintabinv,twidopt);

            %------- Final multiply.
            % y = y(n:m) .* ww(n:m);
            idx = ONE + yoff;
            if isInverse
                denom = cast(nRows,'like',x);
                for k = coder.internal.indexInt(n):coder.internal.indexInt(m)
                    y(idx) = coder.internal.conjtimes(wwc(k), fv(k));
                    y(idx) = y(idx)/denom;
                    idx = idx+1;
                end
            else
                for k = coder.internal.indexInt(n):coder.internal.indexInt(m)
                    % y(idx) = fv(k) * ww(k);
                    y(idx) = coder.internal.conjtimes(wwc(k), fv(k));
                    idx = idx+1;
                end
            end
        end
        function useRadix2 = get_size_props(this,algid, nfft)
            coder.internal.prefer_const(algid, nfft);
            if algid == this.AUTO
                useRadix2 = coder.internal.sizeIsPow2(nfft);
            elseif algid == this.RADIX2
                coder.internal.assert(coder.internal.sizeIsPow2(nfft), 'Coder:toolbox:eml_fft_sizeMustBePower2');
                useRadix2 = true;
            else % if algid == BLUESTEIN
                useRadix2 = false;
            end
        end
        function [n2blue, nRows] = get_algo_sizes(~,nfft, useRadix2)
            coder.internal.prefer_const(nfft, useRadix2);
            ONE = coder.internal.indexInt(1);
            n2blue = ONE;
            if useRadix2
                nRows = coder.internal.indexInt(nfft);
            else
                if nfft > 0
                    nn1m1 = nfft+nfft-1;
                    n2blue = bitshift(ONE,nextpow2(nn1m1));
                else
                    n2blue = ONE;
                end
                assert(n2blue <= eml_max(4*nfft,1)); %<HINT>
                nRows = coder.internal.indexInt(n2blue);
            end
        end
        function [costab, sintab, sintabinv] = generate_twiddle_tables(this,x, isInverse, nRows, twidopt, useRadix2)
            coder.internal.prefer_const(isInverse, nRows, twidopt, useRadix2);
            e = 2*pi/cast(nRows,'like',real(x));
            TWO = coder.internal.indexInt(2);
            nRowsD2 = coder.internal.indexDivide(nRows,TWO);
            nRowsD4 = coder.internal.indexDivide(nRowsD2,TWO);

            % Generate twiddle table if desired.
            if coder.const(twidopt ~= this.NO_TWIDDLE_TABLE)
                costab1q = this.make_1q_cosine_table(e,nRowsD4);
                if coder.const(twidopt == this.FULL_TWIDDLE_TABLE)
                    if ~useRadix2
                        % Bluestein needs both forward and inverse FFTs
                        [costab,sintab,sintabinv] = this.make_twiddle_table(costab1q,true);
                        if coder.internal.isConst(costab1q)
                            costab = coder.const(costab);
                            sintab = coder.const(sintab);
                            sintabinv = coder.const(sintabinv);
                        end
                    else
                        [costab,sintab] = this.make_twiddle_table(costab1q,isInverse);
                        if coder.internal.isConst(costab1q)
                            costab = coder.const(costab);
                            sintab = coder.const(sintab);
                        end
                        % Use empty row to prevent copy
                        sintabinv = zeros(1,0,'like',sintab);
                    end
                else
                    costab = costab1q;
                    sintab = [];
                    sintabinv = [];
                end
            else
                costab = [];
                sintab = [];
                sintabinv = [];
            end

        end
        function bitrevIndex = get_bitrevIndex(this,nRowsM1,nfftLen)
        % Get bit reversed index. Store for use in fft in place algorithm
        % Define constants
            ZERO = coder.internal.indexInt(0);
            ONE = coder.internal.indexInt(1);

            ju = zeros(this.ucls);
            iy = ZERO;
            bitrevIndex = zeros(nfftLen,1,coder.internal.indexIntClass());
            for j1 = ONE:nRowsM1
                bitrevIndex(j1) = iy+ONE;
                ju = this.fft_bitrevidx(ju,nfftLen);
                iy = eml_cast(ju,coder.internal.indexIntClass,'wrap');
            end
            bitrevIndex(nRowsM1+1) = iy+ONE;
        end
        function j = fft_bitrevidx(~,j,n)
        % Compute index for bitreverse operation.
            coder.inline('always');
            tst = true;
            while tst
                n = eml_rshift(n,ones(coder.internal.indexIntClass));
                j = eml_bitxor(j,n);
                tst = eml_bitand(j,n) == 0;
            end
        end
        function [nxeven,minX] = padOrTruncate(this,x,n1)
        % Determine to pad or truncate for the half length signal
        % minX - gives the appropriate index for pad/truncate.
        % nxeven - flag to check if x length is even or if x length is
        % odd and x length is greater than nfft.
            if bitand(cast(size(x,1),this.ucls),ones(this.ucls)) == 0
                nxeven = true;
                sizeX = coder.internal.indexInt(size(x,1));
            else
                if size(x,1) >= n1
                    nxeven = true;
                    sizeX = n1;
                else
                    nxeven = false;
                    sizeX = coder.internal.indexInt(size(x,1) - 1);
                end
            end
            % Helper variable to pad/truncate required elements for fft computation
            minX = eml_min(sizeX,n1);
        end
        function [nRowsM1,nRowsM2,nRowsD2,nRowsD4,e] = radix2_constants(~,y,xType,nRows)
        % Defining constants required for radix2 algorithm
            TWO = coder.internal.indexInt(2);
            SZ1 = coder.internal.indexInt(size(y,1));
            nRowsM1 = eml_min(SZ1,nRows) - 1;
            nRowsM2 = nRows - TWO;
            nRowsD2 = coder.internal.indexDivide(nRows,TWO);
            nRowsD4 = coder.internal.indexDivide(nRowsD2,TWO);
            e = 2*pi/cast(nRows,'like',xType);
        end
        function wrapIndex = calculate_wrapIndex(~,nfftLen)
        % Calculates the wrap around index which is used during
        % reconstruction of the N point fft from the N/2 point fft
        % Define constants
            ONE = coder.internal.indexInt(1);
            TWO = coder.internal.indexInt(2);

            % Compute wrap around index
            wrapIndex = coder.nullcopy(zeros(1,nfftLen,coder.internal.indexIntClass()));
            for i = ONE:nfftLen
                if i ~= ONE
                    wrapIndex(i) = nfftLen-i+TWO;
                else
                    wrapIndex(i) = ONE;
                end
            end
        end
        function y = getback_radix2_fft(~,y,yoff,reconVar1,reconVar2,wrapIndex,hnRows)
        % Construct N point FFT from the N/2 point FFT
        % The construction is followed after simplification of the
        % following equations (1) - (4):
        %               x1(k) = 0.5*(xk + xk(revInd));                   (1)
        %               x2(k) = -0.5i*(xk - xk(revInd));                 (2)
        % where xk is N/2 point FFT of complex vector constructed from x
        % Reconstruction equations:
        %               G(k) = x1(k) + (Wk_2N * x2(k))                  (3)
        %               G(k+N) = x1(k) - (Wk_2N * x2(k))                (4)
        % where k = 0,1,....N-1 ,Wk_2N is the twiddle factors used for
        % obtaining the N point FFT and G is the N point FFT
            ONE = coder.internal.indexInt(1);
            TWO = coder.internal.indexInt(2);
            iterVar = coder.internal.indexDivide(hnRows,TWO);

            % In-place reconstruction for the first element
            i = ONE;
            temp1 = y(yoff+i);
            y(yoff+i) = 0.5*(temp1*reconVar1(i) + conj(temp1)*reconVar2(i));
            y(yoff+hnRows+i) = 0.5*(temp1*reconVar2(i) + conj(temp1)*reconVar1(i));
            % In-place reconstruction for all elements except the last
            % element
            for i = TWO:iterVar
                temp1 = y(yoff+i);
                temp2 = y(yoff+wrapIndex(i));
                y(yoff+i) = 0.5*(temp1*reconVar1(i) + conj(temp2)*reconVar2(i));
                y(yoff+hnRows+i) = 0.5*(temp1*reconVar2(i) + conj(temp2)*reconVar1(i));

                y(yoff+wrapIndex(i)) = 0.5*(temp2*reconVar1(wrapIndex(i)) + conj(temp1)*reconVar2(wrapIndex(i)));
                y(yoff+wrapIndex(i)+hnRows) = 0.5*(temp2*reconVar2(wrapIndex(i)) + conj(temp1)*reconVar1(wrapIndex(i)));
            end
            % In-place reconstruction for the last element
            % In case the input is size 2, then this computation is not
            % required.
            if iterVar ~= 0
                i = iterVar + ONE;
                temp1 = y(yoff+i);
                y(yoff+i) = 0.5*(temp1*reconVar1(i) + conj(temp1)*reconVar2(i));
                y(yoff+hnRows+i) = 0.5*(temp1*reconVar2(i) + conj(temp1)*reconVar1(i));
            end
        end
        function y = getback_bluestein_fft(~,y,yk,yoff,ytmpoff,reconVar1,reconVar2,wrapIndex,hnRows)
        % Construct N point FFT from the N/2 point FFT
        % The construction is followed after simplification of the
        % following equations (1) - (4):
        %               x1(k) = 0.5*(xk + xk(revInd));                   (1)
        %               x2(k) = -0.5i*(xk - xk(revInd));                 (2)
        % where xk is N/2 point FFT of complex vector constructed from x
        % Reconstruction equations:
        %               G(k) = x1(k) + (Wk_2N * x2(k))                  (3)
        %               G(k+N) = x1(k) - (Wk_2N * x2(k))                (4)
        % where k = 0,1,....N-1 ,Wk_2N is the twiddle factors used for
        % obtaining the N point FFT and G is the N point FFT
            ONE = coder.internal.indexInt(1);
            for i = ONE:hnRows
                y(yoff+i) = 0.5*(yk(ytmpoff+i)*reconVar1(i) + conj(yk(ytmpoff+wrapIndex(i)))*reconVar2(i));
                y(yoff+hnRows+i) = 0.5*(yk(ytmpoff+i)*reconVar2(i) + conj(yk(ytmpoff+wrapIndex(i)))*reconVar1(i));
            end
        end
        function [hcostab,hsintab,hcostabinv,hsintabinv] = get_half_twiddle_tables(this,x,algo, ...
                                                                                        costab,sintab,costabinv,sintabinv)
            coder.internal.prefer_const(costab,sintab,x);
            % Define constants
            ONE = coder.internal.indexInt(1);
            TWO = coder.internal.indexInt(2);
            szCostab = coder.internal.indexInt(size(costab,2));
            hszCostab = coder.internal.indexDivide(szCostab,TWO);

            % Twiddle factor computations for performing N/2 point FFT
            hcostab = coder.nullcopy(zeros(1,hszCostab,'like',real(x)));
            hsintab = coder.nullcopy(zeros(1,hszCostab,'like',real(x)));
            if algo == this.BLUESTEIN
                hcostabinv = coder.nullcopy(zeros(1,hszCostab,'like',real(x)));
                hsintabinv = coder.nullcopy(zeros(1,hszCostab,'like',real(x)));
                for i = ONE:hszCostab
                    hcostab(i) = costab(2*i-1);
                    hsintab(i) = sintab(2*i-1);
                    hcostabinv(i) = costabinv(2*i-1);
                    hsintabinv(i) = sintabinv(2*i-1);
                end
            else
                hcostabinv = zeros(1,0,'like',real(x));
                hsintabinv = zeros(1,0,'like',real(x));
                for i = ONE:hszCostab
                    hcostab(i) = costab(2*i-1);
                    hsintab(i) = sintab(2*i-1);
                end
            end
        end
        function [reconVar1,reconVar2] = get_reconstruct_factors(this,algo,x,hnRows, ...
                                                                      costable,sintable,isInverse)
            % Define constants
            ONE = coder.internal.indexInt(1);
            TWO = coder.internal.indexInt(2);

            % Preallocation
            reconVar1 = coder.nullcopy(complex(zeros(hnRows,1,'like',real(x))));
            reconVar2 = coder.nullcopy(complex(zeros(hnRows,1,'like',real(x))));
            if algo == this.BLUESTEIN
                idx = ONE;
                for i = ONE:hnRows
                    if ~isInverse
                        reconVar1(i) = complex(1+sintable(idx),-costable(idx));
                        reconVar2(i) = complex(1-sintable(idx),costable(idx));
                    else
                        reconVar1(i) = complex(1-sintable(idx),-costable(idx));
                        reconVar2(i) = complex(1+sintable(idx),costable(idx));
                    end
                    idx = idx + TWO;
                end
            else  % RADIX2
                for i = ONE:hnRows
                    reconVar1(i) = complex(1+sintable(i),-costable(i));
                    reconVar2(i) = complex(1-sintable(i),costable(i));
                end
            end
        end
        function y = radix2Algo(this,y,chanStart,twidopt,costab,sintab,nRows,nRowsM2,nRowsD2,nRowsD4,e,isInverse)
            coder.inline('always');
            % Define constants
            ZERO = coder.internal.indexInt(0);
            TWO = coder.internal.indexInt(2);

            if twidopt == this.SMALL_TWIDDLE_TABLE
                costab1q = costab;
            end

            % In-place radix-2 decimation-in-time FFT.
            i1 = chanStart;
            i2 = i1+nRowsM2;
            if nRows > 1 % See G313314
                for i = i1:TWO:i2
                    temp = y(i+2);
                    y(i+2) = y(i+1) - temp;
                    y(i+1) = y(i+1) + temp;
                end
            end
            iDelta = TWO;
            iDelta2 = iDelta+iDelta;
            k = nRowsD4;
            iheight = 1 + (k-1)*iDelta2;
            while k > 0
                istart = chanStart;
                j = ZERO;
                % Perform the first butterfly of this stage.  Since twid = 1+0i, no
                % multiplication is required.
                i = istart;
                ihi = i+iheight;
                while i < ihi
                    temp = y(i+iDelta+1);
                    y(i+iDelta+1) = ...
                        y(i+1) - temp;
                    y(i+1) = y(i+1) + temp;
                    i = i+iDelta2;
                end
                % Perform the remaining butterflies of this stage.
                istart = istart+1;
                j = j+k;
                if coder.const(twidopt == this.SMALL_TWIDDLE_TABLE)
                    % We are using a one-quadrant cosine table, so this is split
                    % into two loops in order to extract twid = cos(j*e) +
                    % sin(j*e)*1i from the table without adding logic in the loop.
                    while j < nRowsD4
                        if isInverse
                            twid = complex( ...
                                costab1q(j+1), ...
                                costab1q((nRowsD4-j)+1));
                        else
                            twid = complex( ...
                                costab1q(j+1), ...
                                -costab1q((nRowsD4-j)+1));
                        end
                        i = istart;
                        ihi = i+iheight;
                        while i < ihi
                            temp = twid*y(i+iDelta+1);
                            y(i+iDelta+1) = ...
                                y(i+1) - temp;
                            y(i+1) = y(i+1) + temp;
                            i = i+iDelta2;
                        end
                        istart = istart+1;
                        j = j+k;
                    end
                end
                while j < nRowsD2
                    if coder.const(twidopt == this.FULL_TWIDDLE_TABLE)
                        twid = complex( ...
                            costab(j+1), ...
                            sintab(j+1));
                    elseif coder.const(twidopt == this.SMALL_TWIDDLE_TABLE)
                        if isInverse
                            twid = complex( ...
                                -costab1q((nRowsD2-j)+1), ...
                                costab1q((j-nRowsD4)+1));
                        else
                            twid = complex( ...
                                -costab1q((nRowsD2-j)+1), ...
                                -costab1q((j-nRowsD4)+1));
                        end
                    else
                        theta = e*cast(j,'like',real(e));
                        if isInverse
                            twid = complex(cos(theta),sin(theta));
                        else
                            twid = complex(cos(theta),-sin(theta));
                        end
                    end
                    i = istart;
                    ihi = i+iheight;
                    while i < ihi
                        temp = twid*y(i+iDelta+1);
                        y(i+iDelta+1) = ...
                            y(i+1) - temp;
                        y(i+1) = y(i+1) + temp;
                        i = i+iDelta2;
                    end
                    istart = istart+1;
                    j = j+k;
                end
                k = coder.internal.indexDivide(k,TWO);
                iDelta = iDelta2;
                iDelta2 = iDelta+iDelta;
                iheight = iheight-iDelta;
            end
        end
        function c = ucls(~)
            c = coder.internal.unsignedClass(coder.internal.indexIntClass);

        end
        function costab1q = make_1q_cosine_table(~,e,n)
        % First-quadrant cosine table: costab = cos(e*(0:n)).
        % This function has some tweaks to improve accuracy.
            coder.inline('always');
            coder.internal.prefer_const(e,n);
            costab1q = coder.nullcopy(zeros(1,n+1,'like',real(e)));
            costab1q(1) = 1;
            nd2 = coder.internal.indexDivide(n,2);
            for k = 1:nd2
                costab1q(k+1) = cos(e*cast(k,'like',real(e)));
            end
            for k = nd2+1:n-1
                costab1q(k+1) = sin(e*cast(n-k,'like',real(e)));
            end
            costab1q(n+1) = 0;

        end
        function [costab,sintab,sintabinv] = make_twiddle_table(~,costab1q,isInverse)
        % Generates a full table of complex twiddles from the one-quadrant cosine
        % table.  The full table takes more memory but results in better
        % performance in most cases.
        %
        % Don't return a separate costabinv since it is the same as costab. Callers
        % should resuse costab
            coder.inline('always');
            coder.internal.prefer_const(costab1q,isInverse);
            genBoth = nargout == 3;
            n = cast(size(costab1q,2)-1,coder.internal.indexIntClass);
            n2 = 2*n;
            N = n2+1;
            costab = coder.nullcopy(zeros(1,N,'like',real(costab1q)));
            sintab = coder.nullcopy(zeros(1,N,'like',real(costab1q)));
            costab(1) = 1;
            sintab(1) = 0;
            if genBoth
                sintabinv = coder.nullcopy(zeros(1,N,'like',real(costab1q)));
                for k = 1:n
                    sintabinv(k+1) = costab1q((n-k)+1);
                end
                for k = n+1:n2
                    sintabinv(k+1) = costab1q((k-n)+1);
                end
                for k = 1:n
                    costab(k+1) = costab1q(k+1);
                    sintab(k+1) = -costab1q((n-k)+1);
                end
                for k = n+1:n2
                    costab(k+1) = -costab1q((n2-k)+1);
                    sintab(k+1) = -costab1q((k-n)+1);
                end
            else
                if isInverse
                    for k = 1:n
                        costab(k+1) = costab1q(k+1);
                        sintab(k+1) = costab1q((n-k)+1);
                    end
                    for k = n+1:n2
                        costab(k+1) = -costab1q((n2-k)+1);
                        sintab(k+1) = costab1q((k-n)+1);
                    end
                else
                    for k = 1:n
                        costab(k+1) = costab1q(k+1);
                        sintab(k+1) = -costab1q((n-k)+1);
                    end
                    for k = n+1:n2
                        costab(k+1) = -costab1q((n2-k)+1);
                        sintab(k+1) = -costab1q((k-n)+1);
                    end
                end
            end

        end
        function y = RADIX2(~)
            coder.inline('always');
            y = uint8(0);

        end
        function y = BLUESTEIN(~)
            coder.inline('always');
            y = uint8(1);

        end
        function y = AUTO(~)
            coder.inline('always');
            y = uint8(2);

        end
        function y = NO_TWIDDLE_TABLE(~)
            coder.inline('always');
            y = uint8(0);

        end
        function y = SMALL_TWIDDLE_TABLE(~)
            coder.inline('always');
            y = uint8(1);

        end
        function y = FULL_TWIDDLE_TABLE(~)
            coder.inline('always');
            y = uint8(2);

        end
    end
end

% LocalWords:  Bluestein bluestein nfft Bitreversed bitreversed fy fv ww algid
% LocalWords:  bitreverse nxeven xk Preallocation twid costab costabinv resuse
