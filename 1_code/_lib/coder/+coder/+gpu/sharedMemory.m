function sharedMemory(symbol, varargin)

%   coder.gpu.sharedMemory Pragma for GPU coder shared memory
%   coder.gpu.sharedMemory pragma enables collaborative load and read of an array 
%   by block-threads into shared memory for faster memory access in GPU kernels.  
%   This is particularly useful for sliding window operations, where successive
%   threads read from sliding and overlapping regions of the input array.
% 
%   coder.gpu.sharedMemory(VAR, [Bx, Rx]) loads the 1-D variable VAR into a 1-D 
%   shared memory variable.
%   coder.gpu.sharedMemory(VAR, [Bx, Rx], [By, Ry]) loads the 2-D variable VAR 
%   into a 2-D shared memory variable.
% 
%   VAR is the array variable to be loaded into shared memory.
%   [Bx, Rx] is the starting offset and number of elements to be loaded into
%   shared memory by each thread.
%   For 2-D VAR, [Bx, Rx] and [By, Ry] are the starting offset and number
%   of elements in the x(height) dimension and y(width) dimension.
%
%   Each thread will load Bix:Bix+Rx (and Biy:Biy+Ry) consecutive elements from 
%   VAR into shared memory. 
%   Bix is computed from the offset Bx. Bix = threadIdx + Bx.
%   Rx is a the number of elements loaded by each thread. It is a positive non-zero 
%   constant. 
%
%   Example : 1D filtering operation where Im = rand(1,10)
% 
%   function Op =  test(Im)
%     IW = numel(Im);
%     Op = coder.nullcopy(Im);
%     filter = [-2 ,0, 2];
%     coder.gpu.kernel();
%     for ocol = 1:IW
%   	  iw = ocol - floor((3-1)/2); % center window
% 	      coder.gpu.sharedMemory(Im,[iw, 3]);
%         cval = 0;
%         for c = 1:3	
% 	      idxw = iw + c - 1;
% 	      if ((idxw >= 1) && (idxw <= IW))
% 	          cval = cval + filter(1,c) * Im(1,idxw);
% 	      end
% 	  end
% 	  Op(ocol) = cval;
%     end
%
%   See also GPUCODER.STENCILKERNEL
    
%   Copyright 2016-2017 The MathWorks, Inc.

%#codegen    
  coder.gpu.internal.sharedMemoryImpl(symbol, true, coder.isRowMajor, varargin{:});
end
