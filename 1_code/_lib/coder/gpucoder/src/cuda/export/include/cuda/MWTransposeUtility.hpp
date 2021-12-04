#ifndef __MW_TRANSPOSE_UTIL_H__
#define __MW_TRANSPOSE_UTIL_H__

#ifdef __CUDACC__

#define MW_DEVICE __device__
#define TILE_SIZE 32
#define BLOCKDIM_X 32
#define BLOCKDIM_Y 8

template <typename T>
__global__ void transposeKernel2DColMajor(const T *idata, T* odata, int nrows, int ncols) {
    __shared__ T tile[TILE_SIZE][TILE_SIZE+1];
    
    // 2D coordinates (x, y) => (rows, cols)
    int x = blockIdx.x * TILE_SIZE + threadIdx.x; //rows
    int y = blockIdx.y * TILE_SIZE + threadIdx.y; //cols

    for (int j = 0; j < TILE_SIZE; j += BLOCKDIM_Y) {
        if (y+j < ncols && x < nrows) {
            tile[threadIdx.x][threadIdx.y + j] = idata[(y+j)*nrows + x];;
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    for (int j = 0; j < TILE_SIZE; j += BLOCKDIM_Y) {
        if (y+j < nrows && x < ncols) {
            odata[(y+j)*ncols + x] = tile[threadIdx.y + j][threadIdx.x];
        }
    }
}

template <typename T>
__global__ void transposeKernel2DRowMajor(const T *idata, T* odata, int nrows, int ncols) {
    __shared__ T tile[TILE_SIZE][TILE_SIZE+1];
    
    // 2D coordinates (y, x) => (rows, cols)
    int x = blockIdx.x * TILE_SIZE + threadIdx.x; //cols
    int y = blockIdx.y * TILE_SIZE + threadIdx.y; //rows

    for (int j = 0; j < TILE_SIZE; j += BLOCKDIM_Y) {
        if (y+j < nrows && x < ncols) {
            tile[threadIdx.y + j][threadIdx.x] = idata[(y+j)*ncols + x];;
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    for (int j = 0; j < TILE_SIZE; j += BLOCKDIM_Y) {
        if (y+j < ncols && x < nrows) {
            odata[(y+j)*nrows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

template <class T>
void transposeImpl(const T* src, T* dest, int nrows, int ncols, bool isRowMajor) {

    if(nrows == 1 || ncols == 1) {
        // scalar or 1-d vector
        cudaMemcpy(dest, src, sizeof(T)*nrows*ncols, cudaMemcpyDeviceToDevice);
    } else {
        // 2d matrix
        if(isRowMajor) {
            dim3 dimGrid((ncols-1)/TILE_SIZE + 1,(nrows-1)/TILE_SIZE + 1);
            dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y, 1);
            transposeKernel2DRowMajor<<<dimGrid, dimBlock>>>(src, dest, nrows, ncols);
        } else {
            dim3 dimGrid((nrows-1)/TILE_SIZE + 1,(ncols-1)/TILE_SIZE + 1);
            dim3 dimBlock(BLOCKDIM_X, BLOCKDIM_Y, 1);
            transposeKernel2DColMajor<<<dimGrid, dimBlock>>>(src, dest, nrows, ncols);
        }
    }
}

template <class T> MW_DEVICE
void transposeImplDevice(const T* src, T* dest, int nrows, int ncols, bool isRowMajor) {

    if(nrows == 1 || ncols == 1) {
        // scalar or 1-d vector
        memcpy(dest, src, sizeof(T)*nrows*ncols);
    } else {
        // 2d matrix
        if(isRowMajor) {
            for(int rowid = 0; rowid < nrows; rowid++) {
                for(int colid = 0; colid < ncols; colid++) {
                    dest[colid*nrows + rowid] = src[rowid*ncols + colid];
                }
            }
        } else {
            for(int colid = 0; colid < ncols; colid++) {
                for(int rowid = 0; rowid < nrows; rowid++) {
                    dest[rowid*ncols + colid] = src[colid*nrows + rowid];
                }
            }
        }
    }
}

#endif

#endif
