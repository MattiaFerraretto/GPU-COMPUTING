
#include "cudalinalg.cuh"

extern float DEVICE_TOT_TIME;

/**
 * See the documentation of cuda_ndarrayHost function in the file cudalinalg.cuh 
*/
__host__ ndarray* cuda_ndarrayHost(int rows, int columns)
{
    ndarray* a = (ndarray*)malloc(sizeof(ndarray));
    int* shape = (int*)calloc(2, sizeof(int));

    shape[0] = rows;
    shape[1] = columns;

    a->shape = shape;

    CUDA_CHECK(cudaMallocHost((void**)&a->data, rows * columns * sizeof(float))); 
    
    return a;
}

/**
 * See the documentation of cudaFreeHost_ function in the file cudalinalg.cuh 
*/
__host__ void cudaFreeHost_(ndarray* A)
{
    free(A->shape);
    CUDA_CHECK(cudaFreeHost(A->data));
    free(A);
}

/**
 * See the documentation of cuda_ndarray function in the file cudalinalg.cuh 
*/
__host__ ndarray* cuda_ndarray(int rows, int columns)
{
    ndarray* A_dev = (ndarray*) malloc(sizeof(ndarray));
    int* shape = (int*) malloc(2 * sizeof(int));

    shape[0] = rows;
    shape[1] = columns;

    int n = rows * columns;
    
    CUDA_CHECK(cudaMalloc((void**)&(A_dev->shape), 2 * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&(A_dev->data), n * sizeof(float)));

    //CUDA_CHECK(cudaMemset((void*)(A_dev->data), 0, n * sizeof(double)));

    CUDA_CHECK(cudaMemcpy((void*)(A_dev->shape), (void*)(shape), 2 * sizeof(int), cudaMemcpyHostToDevice));
    free(shape);
    
    return A_dev;
}

/**
 * See the documentation of cudaFree_ function in the file cudalinalg.cuh 
*/
__host__ void cudaFree_(ndarray* A_dev)
{
    CUDA_CHECK(cudaFree(A_dev->shape));
    CUDA_CHECK(cudaFree(A_dev->data));
    free(A_dev);
}

/**
 * See the documentation of memTileH2DAsync function in the file cudalinalg.cuh 
*/
__host__ void memTileH2DAsync(ndarray* dest, ndarray* src, int offset, int dimTile, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync((void*) &(dest->data[offset]), (void*) &(src->data[offset]), dimTile * sizeof(float), cudaMemcpyHostToDevice, stream));
}

/**
 * See the documentation of memTileD2HAsync function in the file cudalinalg.cuh 
*/
__host__ void memTileD2HAsync(ndarray* dest, ndarray* src, int offset, int dimTile, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync((void*) &(dest->data[offset]), (void*) &(src->data[offset]), dimTile * sizeof(float), cudaMemcpyDeviceToHost, stream));
}

/**
 * See the documentation of rowTileH2DAsync function in the file cudalinalg.cuh 
*/
__host__ void rowTileH2DAsync(ndarray* dest, ndarray* src, int offset, int dimTile, cudaStream_t stream)
{
    int pitch = src->shape[1] * sizeof(float);
    int rows = dimTile;
    int columns = src->shape[1] * sizeof(float);

    CUDA_CHECK(cudaMemcpy2DAsync((void*)&dest->data[offset], pitch, (void*)&src->data[offset], pitch, columns , rows, cudaMemcpyHostToDevice, stream));
}

/**
 * See the documentation of rowTileD2HAsync function in the file cudalinalg.cuh 
*/
__host__ void rowTileD2HAsync(ndarray* dest, ndarray* src, int offset, int dimTile, cudaStream_t stream)
{
    int pitch = dest->shape[1] * sizeof(float);
    int rows = dimTile;
    int columns = dest->shape[1] * sizeof(float);

    CUDA_CHECK(cudaMemcpy2DAsync((void*)&dest->data[offset], pitch, (void*)&src->data[offset], pitch, columns , rows, cudaMemcpyDeviceToHost, stream));
}

/**
 * See the documentation of columnTileH2DAsync function in the file cudalinalg.cuh 
*/
__host__ void columnTileH2DAsync(ndarray* dest, ndarray* src, int offset, int dimTile, cudaStream_t stream)
{
    int pitch = src->shape[1] * sizeof(float);
    int rows = src->shape[0];
    int columns = dimTile * sizeof(float);

    CUDA_CHECK(cudaMemcpy2DAsync((void*)&dest->data[offset], pitch, (void*)&src->data[offset], pitch, columns , rows, cudaMemcpyHostToDevice, stream));
}

/**
 * See the documentation of columnTileD2HAsync function in the file cudalinalg.cuh 
*/
__host__ void columnTileD2HAsync(ndarray* dest, ndarray* src, int offset, int dimTile, cudaStream_t stream)
{
    int pitch = dest->shape[1] * sizeof(float);
    int rows = dest->shape[0];
    int columns = dimTile * sizeof(float);

    CUDA_CHECK(cudaMemcpy2DAsync((void*)&dest->data[offset], pitch, (void*)&src->data[offset], pitch, columns , rows, cudaMemcpyDeviceToHost, stream));
}




/**
 * See the documentation of cudaVSDivision kernel in the file cudalinalg.cuh 
*/
__global__ void cudaVSDivision(ndarray A, ndarray C, float value, int offset, int dimTile)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    int ofst = offset;
    float scalar = value;
    __shared__ float tile[BLOCKDIMX * BLOCKDIMY];
    
    if(i >= dimTile)
        return ;

    tile[tid] = A.data[i + ofst];
    __syncthreads();

    C.data[i + ofst] = tile[tid] / scalar;
}

/**
 * See the documentation of cudaVSDivision function in the file cudalinalg.cuh 
*/
__host__ ndarray* cudaVSDivision(ndarray* A, float value, int nTile, bool inPlace, bool verbose)
{
    if(A->shape[0] > 1 && A->shape[1] > 1)
    {
        printf("ERROR:: File: %s, Line: %d, Function name: cudaVSDivision, ", __FILE__, __LINE__);
        printf("reason: (%d, %d) != (1, n) || (%d, %d) != (n, 1); A must be a row vector (1, n) or a column vector (n, 1).\n", A->shape[0], A->shape[1], A->shape[0], A->shape[1]);
        exit(EXIT_FAILURE); 
    }

    ndarray* C = !inPlace ? cuda_ndarrayHost(A->shape[0], A->shape[1]) : A;

    ndarray* A_dev = A->shape[0] > 1 ?  cuda_ndarray(A->shape[1], A->shape[0]) : cuda_ndarray(A->shape[0], A->shape[1]);
    ndarray* C_dev = !inPlace ? cuda_ndarray(A->shape[0], A->shape[1]) : A_dev;

    int lenght = A->shape[0] > 1 ? A->shape[0] : A->shape[1]; 
    int dimTile = (lenght % nTile == 0) ? lenght / nTile : (lenght + nTile) / nTile;
    int nStreams = (lenght % dimTile == 0) ?lenght / dimTile : (lenght + dimTile) / dimTile;
    
    cudaStream_t* streams = (cudaStream_t*) malloc(nStreams * sizeof(cudaStream_t));
    cudaEvent_t* startEvents = (cudaEvent_t*) malloc(nStreams * sizeof(cudaEvent_t));
    cudaEvent_t* stopEvents = (cudaEvent_t*) malloc(nStreams * sizeof(cudaEvent_t));

    dim3 block(BLOCKDIMX * BLOCKDIMY);

    if(verbose)
    {   
        int coeff = inPlace ? 1 : 2;
        printf("----------------------------------- START cudaVSDivision -----------------------------------\n");
        printf("Allocated memory for A[%d] (GB): %.4f\n", lenght, (lenght * sizeof(float)) / (double) pow(2, 30));
        printf("Total host/device allocated memory (GB): %.4f\n", (lenght * coeff *  sizeof(float)) / (double) pow(2, 30));
        printf("Streams: %d\n", nStreams);
    }

    for(int i = 0; i < nStreams; i++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaEventCreate(&startEvents[i]));
        CUDA_CHECK(cudaEventCreate(&stopEvents[i]));

        int offset = i * dimTile;

        if(offset + dimTile > lenght)  dimTile = dimTile - ((offset + dimTile) - lenght);

        dim3 grid((dimTile + block.x - 1) / block.x);

        if(verbose)
            printf("cudaVSDivision%d <<< (%d, %d), (%d, %d), 0, %d >>>, dimTile: %d, offset: %d, lenght: %d\n", i, grid.x, grid.y, block.x, block.y, i + 1, dimTile, offset,lenght);

        CUDA_CHECK(cudaEventRecord(startEvents[i], streams[i]));

        memTileH2DAsync(A_dev, A, offset, dimTile, streams[i]);

        cudaVSDivision <<< grid, block, 0, streams[i] >>> (*A_dev, *C_dev, value, offset, dimTile);

        memTileD2HAsync(C, C_dev, offset, dimTile, streams[i]);

        CUDA_CHECK(cudaEventRecord(stopEvents[i], streams[i]));

    }

    CUDA_CHECK(cudaDeviceSynchronize());

    float elapsedTime = 0.f;
    for(int i = 0; i < nStreams; i++)
    {
        float millsec;
        CUDA_CHECK(cudaEventElapsedTime(&millsec, startEvents[i], stopEvents[i]));

        elapsedTime += millsec; 

        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaEventDestroy(startEvents[i]));
        CUDA_CHECK(cudaEventDestroy(stopEvents[i]));
    }
    
    DEVICE_TOT_TIME += elapsedTime;

    if(verbose) 
    {
        printf("cudaVSDivision's time (s): %.6f\n", elapsedTime / 1000);
        printf("----------------------------------- FINISH cudaVSDivision ----------------------------------\n");
    }

    cudaFree_(A_dev);
    if(!inPlace) cudaFree_(C_dev);
    free(streams);
    free(startEvents);
    free(stopEvents);

    return inPlace ? NULL : C;
}



/**
 * See the documentation of cudaEDistance kernel in the file cudalinalg.cuh 
*/
__global__ void cudaEDistance(ndarray A, ndarray B, ndarray C, int offset, int dimTile, int coffset)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    int ofst = offset;
    int cofst = coffset;

    __shared__ float As[BLOCKDIMX * BLOCKDIMY];
    __shared__ float Bs[BLOCKDIMX * BLOCKDIMY];


    if(i < dimTile)
    {
        As[tid] = A.data[i + ofst];
        Bs[tid] = B.data[i + ofst];
    }
    else
    {
        As[tid] = 0.f;
        Bs[tid] = 0.f;
    }
    __syncthreads();

    As[tid] = (As[tid] - Bs[tid]) * (As[tid] - Bs[tid]);
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {   
            As[tid] += As[tid + stride];
        }
        __syncthreads();
    }

    if(tid == 0)
        C.data[cofst + blockIdx.x] = As[0];
    
}

/**
 * See the documentation of cudaEDistance function in the file cudalinalg.cuh 
*/
__host__ float cudaEDistance(ndarray* A, ndarray* B, int nTile, bool verbose)
{
    // (n, 1) (1, n)
    // (1, n) (n, 1)
    // (1, n) (1, n)
    // (n, 1) (n, 1)
    bool check1 = A->shape[0] > 1 && !(B->shape[0] > 1) && A->shape[0] != B->shape[1];
    bool check2 = !(A->shape[0] > 1) && B->shape[0] > 1 && A->shape[1] != B->shape[0];
    bool check3 = !(A->shape[0] > 1) && !(B->shape[0] > 1) && A->shape[1] != B->shape[1];
    bool check4 = A->shape[0] > 1 && B->shape[0] > 1 && A->shape[0] != B->shape[0];

    if(check1 || check2 || check3 || check4)
    {
        printf("ERROR:: File: %s, Line: %d, Function name: cudaEDistance, ", __FILE__, __LINE__);
        printf("reason: (%d, %d) != (%d, %d); incompatible shape, A and B must be row vectors (1, n) or column vectors (n, 1).\n", A->shape[0], A->shape[1], B->shape[0], B->shape[1]);
        exit(EXIT_FAILURE); 
    }

    int lenght = A->shape[0] > 1 ? A->shape[0] : A->shape[1]; 
    int dimTile = (lenght % nTile == 0) ? lenght / nTile : (lenght + nTile) / nTile;
    int nStreams = (lenght % dimTile == 0) ?lenght / dimTile : (lenght + dimTile) / dimTile;

    dim3 block(BLOCKDIMX * BLOCKDIMY);

    ndarray* C = cuda_ndarrayHost(1 , nStreams * (( dimTile + block.x - 1) / block.x));

    ndarray* A_dev = A->shape[0] > 1 ? cuda_ndarray(A->shape[1], A->shape[0]) :  cuda_ndarray(A->shape[0], A->shape[1]);
    ndarray* B_dev = B->shape[0] > 1 ? cuda_ndarray(A->shape[1], A->shape[0]) :  cuda_ndarray(B->shape[0], B->shape[1]);
    ndarray* C_dev = cuda_ndarray(C->shape[0], C->shape[1]);

    cudaStream_t* streams = (cudaStream_t*) malloc(nStreams * sizeof(cudaStream_t));
    cudaEvent_t* startEvents = (cudaEvent_t*) malloc(nStreams * sizeof(cudaEvent_t));
    cudaEvent_t* stopEvents = (cudaEvent_t*) malloc(nStreams * sizeof(cudaEvent_t));

    if(verbose)
    {
        printf("----------------------------------- START cudaEDistance -----------------------------------\n");
        printf("Allocated memory for A[%d] (GB): %.4f\n", lenght, (lenght * sizeof(float)) / (double) pow(2, 30));
        printf("Allocated memory for B[%d] (GB): %.4f\n", lenght, (lenght * sizeof(float)) / (double) pow(2, 30));
        printf("Allocated memory for C[%d] (GB): %.4f\n", C->shape[1], (C->shape[1] * sizeof(float)) / (double) pow(2, 30));
        printf("Total host/device allocated memory (GB): %.4f\n", ((lenght * 2 + C->shape[1]) * sizeof(float)) / (double) pow(2, 30));
        printf("Streams: %d\n", nStreams);
    }

    for(int i = 0; i < nStreams; i++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaEventCreate(&startEvents[i]));
        CUDA_CHECK(cudaEventCreate(&stopEvents[i]));

        int offset = i * dimTile;

        if(offset + dimTile > lenght)  dimTile = dimTile - ((offset + dimTile) - lenght);

        dim3 grid((dimTile + block.x - 1) / block.x);

        if(verbose)
            printf("cudaEDistance%d <<< (%d, %d), (%d, %d), 0, %d >>>, dimTile: %d, offset: %d, lenght: %d\n", i, grid.x, grid.y, block.x, block.y, i + 1, dimTile, offset, lenght);

        CUDA_CHECK(cudaEventRecord(startEvents[i], streams[i]));

        memTileH2DAsync(A_dev, A, offset, dimTile, streams[i]);
        memTileH2DAsync(B_dev, B, offset, dimTile, streams[i]);

        cudaEDistance <<< grid, block, 0, streams[i] >>> (*A_dev, *B_dev, *C_dev, offset, dimTile, i * grid.x);

        memTileD2HAsync(C, C_dev, i * grid.x, grid.x, streams[i]);

        CUDA_CHECK(cudaEventRecord(stopEvents[i], streams[i]));

    }

    CUDA_CHECK(cudaDeviceSynchronize());

    
    float elapsedTime = 0.f;
    for(int i = 0; i < nStreams; i++)
    {
        float millsec;
        CUDA_CHECK(cudaEventElapsedTime(&millsec, startEvents[i], stopEvents[i]));

        elapsedTime += millsec;

        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaEventDestroy(startEvents[i]));
        CUDA_CHECK(cudaEventDestroy(stopEvents[i]));
    }

    DEVICE_TOT_TIME += elapsedTime;
    
    if(verbose) 
    {
        printf("cudaEDistance's time (s): %.6f\n", elapsedTime / 1000);
        printf("----------------------------------- FINISH cudaEDistance ----------------------------------\n");
    }

    cudaFree_(A_dev);
    cudaFree_(B_dev);
    cudaFree_(C_dev);
    
    free(streams);
    free(startEvents);
    free(stopEvents);

    double distance = 0.f;
    for(int i = 0; i < C->shape[1]; i++)
    {
        distance += C->data[i];
    }

    cudaFreeHost_(C);

    return distance == 0 ? 0 : sqrt(distance);
}



/**
 * See the documentation of cudaMSProduct kernel in the file cudalinalg.cuh 
*/
__global__ void cudaMSProduct(ndarray A, ndarray C, float value, int rowOffset, int rowsTile)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    int columns = A.shape[1];
    float scalar = value;
    int rf = rowOffset;

    __shared__ float tile[BLOCKDIMX][BLOCKDIMY];


    if(i < rowsTile && j < columns)
        tile[threadIdx.y][threadIdx.x] = A.data[columns * (i + rf) + j];

    __syncthreads();

    if(i < rowsTile && j < columns)
        C.data[columns * (i + rf) + j] = tile[threadIdx.y][threadIdx.x] * scalar;
}

/**
 * See the documentation of cudaMSproduct function in the file cudalinalg.cuh 
*/
__host__ ndarray* cudaMSProduct(ndarray* A, float value, int nTile, bool inPlace, bool verbose)
{
    ndarray* C = !inPlace ? cuda_ndarrayHost(A->shape[0], A->shape[1]) : A;

    ndarray* A_dev = cuda_ndarray(A->shape[0], A->shape[1]);
    ndarray* C_dev = !inPlace ? cuda_ndarray(A->shape[0], A->shape[1]) : A_dev;

    int rowsTile = (A->shape[0] % nTile == 0) ? A->shape[0] / nTile : (A->shape[0] + nTile) / nTile;
    int nStreams = (A->shape[0] % rowsTile == 0) ? A->shape[0] / rowsTile : (A->shape[0] + rowsTile) / rowsTile;

    cudaStream_t* streams = (cudaStream_t*) malloc(nStreams * sizeof(cudaStream_t));
    cudaEvent_t* startEvents = (cudaEvent_t*) malloc(nStreams * sizeof(cudaEvent_t));
    cudaEvent_t* stopEvents = (cudaEvent_t*) malloc(nStreams * sizeof(cudaEvent_t));

    dim3 block(BLOCKDIMX, BLOCKDIMY);

    if(verbose)
    {
        int coeff = inPlace ? 1 : 2;
        printf("----------------------------------- START cudaMSProduct -----------------------------------\n");
        printf("Allocated memory for A[%d][%d] (GB): %.4f\n", A->shape[0], A->shape[1], (A->shape[0] * A->shape[1] * sizeof(float)) / (double) pow(2, 30));
        printf("Total host/device allocated memory (GB): %.4f\n", (C->shape[0] * C->shape[1] * coeff * sizeof(float)) / (double) pow(2, 30));
        printf("Streams: %d\n", nStreams);
    }

    for (int i = 0; i < nStreams; i++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaEventCreate(&startEvents[i]));
        CUDA_CHECK(cudaEventCreate(&stopEvents[i]));
        
        int rowOffest = i * rowsTile;

        if(rowOffest + rowsTile > A->shape[0])  rowsTile = rowsTile - ((rowOffest + rowsTile) - A->shape[0]);

        int offset =  rowOffest * A->shape[1];

        int gridx = rowsTile * A->shape[1] <= block.x * block.y ? 1 : (A->shape[1] + block.x - 1) / block.x;
        int gridy = rowsTile * A->shape[1] <= gridx * block.x * block.y ? 1 : (rowsTile + block.y - 1) / block.y;
        dim3 grid(gridx, gridy);
        
        if(verbose)
            printf("cudaMSProduct%d <<< (%d, %d), (%d, %d), 0, %d >>>, rowOffset: %d, rowsTile: %d, columns: %d, offset: %d\n", i, gridx, gridy, block.x, block.y, i + 1, rowOffest, rowsTile, A->shape[1], offset);

        CUDA_CHECK(cudaEventRecord(startEvents[i], streams[i]));

        rowTileH2DAsync(A_dev, A, offset, rowsTile, streams[i]);

        cudaMSProduct <<< grid, block, 0, streams[i] >>> (*A_dev, *C_dev, value, rowOffest, rowsTile);

        rowTileD2HAsync(C, C_dev, offset, rowsTile, streams[i]);

        CUDA_CHECK(cudaEventRecord(stopEvents[i], streams[i]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    float elapsedTime = 0.f;
    for(int i = 0; i < nStreams; i++)
    {
        float millsec;
        CUDA_CHECK(cudaEventElapsedTime(&millsec, startEvents[i], stopEvents[i]));

        elapsedTime += millsec; 

        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaEventDestroy(startEvents[i]));
        CUDA_CHECK(cudaEventDestroy(stopEvents[i]));
    }

    DEVICE_TOT_TIME += elapsedTime;

    if(verbose) 
    {
        printf("cudaMSProduct's time (s): %.6f\n", elapsedTime / 1000);
        printf("----------------------------------- FINISH cudaMSProduct ----------------------------------\n");
    }

    cudaFree_(A_dev);
    if(!inPlace) cudaFree_(C_dev);
    free(streams);
    free(startEvents);
    free(stopEvents);

    return inPlace ? NULL : C;
}



/**
 * See the documentation of cudaMTranspose kernel in the file cudalinalg.cuh 
*/
__global__ void cudaMTranspose(ndarray A, ndarray AT, int rowOffset, int rowsTile)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    int rf = rowOffset;
    int rows = A.shape[0];
    int columns = A.shape[1];

    __shared__ float tile[BLOCKDIMX][BLOCKDIMY];
    

    if(i < rowsTile && j < columns)
        tile[threadIdx.y][threadIdx.x] = A.data[columns * (i + rf) + j];
    
    __syncthreads();

    if(i < rowsTile && j < columns)
        AT.data[rows * j + i + rf] = tile[threadIdx.y][threadIdx.x];
}

/**
 * See the documentation of cudaMTranspose function in the file cudalinalg.cuh 
*/
__host__ ndarray* cudaMTranspose(ndarray* A, int nTile, bool verbose)
{
    ndarray* AT = cuda_ndarrayHost(A->shape[1], A->shape[0]);

    ndarray* A_dev = cuda_ndarray(A->shape[0], A->shape[1]);
    ndarray* AT_dev = cuda_ndarray(A->shape[1], A->shape[0]);

    int rowsTile = (A->shape[0] % nTile == 0) ? A->shape[0] / nTile : (A->shape[0] + nTile) / nTile;
    int nStreams = (A->shape[0] % rowsTile == 0) ? A->shape[0] / rowsTile : (A->shape[0] + rowsTile) / rowsTile;

    cudaStream_t* streams = (cudaStream_t*) malloc(nStreams * sizeof(cudaStream_t));
    cudaEvent_t* startEvents = (cudaEvent_t*) malloc(nStreams * sizeof(cudaEvent_t));
    cudaEvent_t* stopEvents = (cudaEvent_t*) malloc(nStreams * sizeof(cudaEvent_t));

    dim3 block(BLOCKDIMX, BLOCKDIMY);

    if(verbose)
    {   
        printf("----------------------------------- START cudaMTranspose -----------------------------------\n");
        printf("Allocated memory for A[%d][%d] (GB): %.4f\n", A->shape[0], A->shape[1], (A->shape[0] * A->shape[1] * sizeof(float)) / (double) pow(2, 30));
        printf("Total host/device allocated memory (GB): %.4f\n", (A->shape[0] * A->shape[1] * 2 * sizeof(float)) / (double) pow(2, 30));
        printf("Streams: %d\n", nStreams);
    }

    for (int i = 0; i < nStreams; i++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaEventCreate(&startEvents[i]));
        CUDA_CHECK(cudaEventCreate(&stopEvents[i]));
        
        int rowOffest = i * rowsTile;

        if(rowOffest + rowsTile > A->shape[0])  rowsTile = rowsTile - ((rowOffest + rowsTile) - A->shape[0]);

        int offset =  rowOffest * A->shape[1];

        int gridx = (A->shape[1] + block.x - 1) / block.x;
        int gridy = (rowsTile + block.y - 1) / block.y;
        dim3 grid(gridx, gridy);
        
        if(verbose)
            printf("cudaMTranspose%d <<< (%d, %d), (%d, %d), 0, %d >>>, rowOffset: %d, rowsTile: %d, columns: %d, offset: %d\n", i, gridx, gridy, block.x, block.y, i + 1, rowOffest, rowsTile, A->shape[1], offset);

        CUDA_CHECK(cudaEventRecord(startEvents[i], streams[i]));

        rowTileH2DAsync(A_dev, A, offset, rowsTile, streams[i]);

        cudaMTranspose <<< grid, block, 0, streams[i] >>> (*A_dev, *AT_dev, rowOffest, rowsTile);

        columnTileD2HAsync(AT, AT_dev, rowOffest, rowsTile, streams[i]);

        CUDA_CHECK(cudaEventRecord(stopEvents[i], streams[i]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    float elapsedTime = 0.f;
    for(int i = 0; i < nStreams; i++)
    {
        float millsec;
        CUDA_CHECK(cudaEventElapsedTime(&millsec, startEvents[i], stopEvents[i]));

        elapsedTime += millsec; 

        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaEventDestroy(startEvents[i]));
        CUDA_CHECK(cudaEventDestroy(stopEvents[i]));
    }

    DEVICE_TOT_TIME += elapsedTime;
    
    if(verbose) 
    {
        printf("cudaMTranspose's time (s): %.6f\n", elapsedTime / 1000);
        printf("----------------------------------- FINISH cudaMTranspose ----------------------------------\n");
    }

    cudaFree_(A_dev);
    cudaFree_(AT_dev);
    free(streams);
    free(startEvents);
    free(stopEvents);

    return AT;
}



/**
 * See the documentation of cudaMMSub kernel in the file cudalinalg.cuh 
*/
__global__ void cudaMMSub(ndarray A, ndarray B, ndarray C, int rowOffset, int rowsTile)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    int columns = A.shape[1];
    int rf = rowOffset;

    __shared__ float As[BLOCKDIMX][BLOCKDIMY];
    __shared__ float Bs[BLOCKDIMX][BLOCKDIMY];
    

    if(i < rowsTile  && j < columns)
    {
        As[threadIdx.y][threadIdx.x] = A.data[columns * (i + rf) + j];
        Bs[threadIdx.y][threadIdx.x] = B.data[columns * (i + rf) + j];
    }
    
    __syncthreads();

    if(i < rowsTile  && j < columns)
        C.data[columns * (i + rf) + j] = As[threadIdx.y][threadIdx.x] - Bs[threadIdx.y][threadIdx.x];
}

/**
 * See the documentation of cudaMMSub function in the file cudalinalg.cuh 
*/
__host__ ndarray* cudaMMSub(ndarray* A, ndarray* B, int nTile, bool inPlace, bool verbose)
{
    if(A->shape[0] != B->shape[0] || A->shape[1] != B->shape[1])
    {
        printf("ERROR:: File: %s, Line: %d, Function name: cudaMMSub, ", __FILE__, __LINE__);
        printf("reason: %d != %d || %d != %d; A and B must have the same size.\n", A->shape[0], B->shape[0], A->shape[1], B->shape[1]);
        exit(EXIT_FAILURE); 
    }

    ndarray* C = !inPlace ? cuda_ndarrayHost(A->shape[0], A->shape[1]) : A;

    ndarray* A_dev = cuda_ndarray(A->shape[0], A->shape[1]);
    ndarray* B_dev = cuda_ndarray(B->shape[0], B->shape[1]);
    ndarray* C_dev = !inPlace ? cuda_ndarray(A->shape[0], A->shape[1]) : A_dev;

    int rowsTile = (A->shape[0] % nTile == 0) ? A->shape[0] / nTile : (A->shape[0] + nTile) / nTile;
    int nStreams = (A->shape[0] % rowsTile == 0) ? A->shape[0] / rowsTile : (A->shape[0] + rowsTile) / rowsTile;

    cudaStream_t* streams = (cudaStream_t*) malloc(nStreams * sizeof(cudaStream_t));
    cudaEvent_t* startEvents = (cudaEvent_t*) malloc(nStreams * sizeof(cudaEvent_t));
    cudaEvent_t* stopEvents = (cudaEvent_t*) malloc(nStreams * sizeof(cudaEvent_t));

    dim3 block(BLOCKDIMX, BLOCKDIMY);

    if(verbose)
    {
        int coeff = inPlace ? 2 : 3;
        printf("----------------------------------- START cudaMMSub -----------------------------------\n");
        printf("Allocated memory for A[%d][%d] (GB): %.4f\n", A->shape[0], A->shape[1], (A->shape[0] * A->shape[1] * sizeof(float)) / (double) pow(2, 30));
        printf("Allocated memory for B[%d][%d] (GB): %.4f\n", B->shape[0], B->shape[1], (B->shape[0] * B->shape[1] * sizeof(float)) / (double) pow(2, 30));
        printf("Total host/device allocated memory (GB): %.4f\n", (C->shape[0] * C->shape[1] * coeff * sizeof(float)) / (double) pow(2, 30));
        printf("Streams: %d\n", nStreams);
    }

    for (int i = 0; i < nStreams; i++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaEventCreate(&startEvents[i]));
        CUDA_CHECK(cudaEventCreate(&stopEvents[i]));
        
        int rowOffest = i * rowsTile;

        if(rowOffest + rowsTile > A->shape[0])  rowsTile = rowsTile - ((rowOffest + rowsTile) - A->shape[0]);

        int offset =  rowOffest * A->shape[1];

        int gridx = rowsTile * A->shape[1] <= block.x * block.y ? 1 : (A->shape[1] + block.x - 1) / block.x;
        int gridy = rowsTile * A->shape[1] <= gridx * block.x * block.y ? 1 : (rowsTile + block.y - 1) / block.y;
        dim3 grid(gridx, gridy);
        
        if(verbose)
            printf("cudaMMSub%d <<< (%d, %d), (%d, %d), 0, %d >>>, rowOffset: %d, rowsTile: %d, columns: %d, offset: %d\n", i, gridx, gridy, block.x, block.y, i + 1, rowOffest, rowsTile, A->shape[1], offset);

        CUDA_CHECK(cudaEventRecord(startEvents[i], streams[i]));

        rowTileH2DAsync(A_dev, A, offset, rowsTile, streams[i]);
        rowTileH2DAsync(B_dev, B, offset, rowsTile, streams[i]);

        cudaMMSub <<< grid, block, 0, streams[i] >>> (*A_dev, *B_dev, *C_dev, rowOffest, rowsTile);

        rowTileD2HAsync(C, C_dev, offset, rowsTile, streams[i]);

        CUDA_CHECK(cudaEventRecord(stopEvents[i], streams[i]));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    float elapsedTime = 0.f;
    for(int i = 0; i < nStreams; i++)
    {
        float millsec;
        CUDA_CHECK(cudaEventElapsedTime(&millsec, startEvents[i], stopEvents[i]));

        elapsedTime += millsec; 

        CUDA_CHECK(cudaStreamDestroy(streams[i]));
        CUDA_CHECK(cudaEventDestroy(startEvents[i]));
        CUDA_CHECK(cudaEventDestroy(stopEvents[i]));
    }

    DEVICE_TOT_TIME += elapsedTime;

    if(verbose) 
    {
        printf("cudaMMSub's time (s): %.6f\n", elapsedTime / 1000);
        printf("----------------------------------- FINISH cudaMMSub ----------------------------------\n");
    }

    cudaFree_(A_dev);
    cudaFree_(B_dev);
    if(!inPlace) cudaFree_(C_dev);
    free(streams);
    free(startEvents);
    free(stopEvents);

    return inPlace ? NULL : C;
}



/**
 * See the documentation of cudaMMProduct kernel in the file cudalinalg.cuh 
*/
__global__ void cudaMMProduct (ndarray A, ndarray B, ndarray C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[BLOCKDIMX][BLOCKDIMX];
    __shared__ float Bs[BLOCKDIMX][BLOCKDIMX];

    int aRws = A.shape[0];
    int bCls = B.shape[1];
    int aCls = A.shape[1];

    float sum[4] = {0.f};

    for (int tIdx = 0; tIdx < (aCls + BLOCKDIMX - 1) / BLOCKDIMX; ++tIdx) {
        int r = tIdx * BLOCKDIMX + threadIdx.y;
        int c = tIdx * BLOCKDIMX + threadIdx.x;

        if (row < aRws && c < aCls)
            As[threadIdx.y][threadIdx.x] = A.data[row * aCls + c];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (r < aCls && col < bCls)
            Bs[threadIdx.y][threadIdx.x] = B.data[r * bCls + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCKDIMX; k += 4)
        {
            sum[0] += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            sum[1] += As[threadIdx.y][k + 1] * Bs[k + 1][threadIdx.x];
            sum[2] += As[threadIdx.y][k + 2] * Bs[k + 2][threadIdx.x];
            sum[3] += As[threadIdx.y][k + 3] * Bs[k + 3][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < aRws && col < bCls)
        C.data[row * bCls + col] = sum[0] + sum[1] + sum[2] + sum[3];
}

/**
 * See the documentation of cudaMMProduct function in the file cudalinalg.cuh 
*/
__host__ ndarray* cudaMMProduct(ndarray* A, ndarray* B, bool verbose)
{
    if(A->shape[1] != B->shape[0])
    {
        printf("ERROR:: File: %s, Line: %d, Function name: cudaMMProduct, ", __FILE__, __LINE__);
        printf("reason: %d != %d ; Columns of A must be equals to rows of B.\n", A->shape[1], B->shape[0]);
        exit(EXIT_FAILURE); 
    }

    ndarray* C = cuda_ndarrayHost(A->shape[0], B->shape[1]);

    ndarray* A_dev = cuda_ndarray(A->shape[0], A->shape[1]);
    ndarray* B_dev = cuda_ndarray(B->shape[0], B->shape[1]);
    ndarray* C_dev = cuda_ndarray(A->shape[0], B->shape[1]);

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;

    if(verbose)
    {
        int A_elms = A->shape[0] * A->shape[1];
        int B_elms = B->shape[0] * B->shape[1];
        int C_elms = C->shape[0] * C->shape[1];
        printf("----------------------------------- START cudaMMProduct -----------------------------------\n");
        printf("Allocated memory for A[%d][%d] (GB): %.4f\n", A->shape[0], A->shape[1], (A_elms * sizeof(float)) / (double) pow(2, 30));
        printf("Allocated memory for B[%d][%d] (GB): %.4f\n", B->shape[0], B->shape[1], (B_elms * sizeof(float)) / (double) pow(2, 30));
        printf("Allocated memory for C[%d][%d] (GB): %.4f\n", C->shape[0], C->shape[1], (C_elms * sizeof(float)) / (double) pow(2, 30));
        printf("Total host/device allocated memory (GB): %.4f\n", ((A_elms + B_elms+ C_elms) * sizeof(float)) / (double) pow(2, 30));
    }

    
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    
    dim3 block(BLOCKDIMX, BLOCKDIMY);
    dim3 grid((B->shape[1] + block.y - 1) / block.y, (A->shape[0] + block.x - 1) / block.x);
    
    if(verbose)
        printf("cudaMMProduct <<< (%d, %d), (%d, %d) >>>\n", grid.x, grid.y, block.x, block.y);

    CUDA_CHECK(cudaEventRecord(startEvent));

    rowTileH2DAsync(A_dev, A, 0, A->shape[0], 0);
    rowTileH2DAsync(B_dev, B, 0, B->shape[0], 0);

    cudaMMProduct <<< grid, block >>> (*A_dev, *B_dev, *C_dev);

    rowTileD2HAsync(C, C_dev, 0, C->shape[0], 0);

    CUDA_CHECK(cudaEventRecord(stopEvent));

    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

    DEVICE_TOT_TIME += elapsedTime;

    if(verbose) 
    {
        printf("cudaMMProduct's time (s): %.6f\n", elapsedTime / 1000);
        printf("----------------------------------- FINISH cudaMMProduct ----------------------------------\n");
    }

    cudaFree_(A_dev);
    cudaFree_(B_dev);
    cudaFree_(C_dev);

    return C;
}



/**
 * See the documentation of eigenvectors function in the file cudalinalg.cuh 
*/
__host__ ndarray* cudaEigenvectors(ndarray* M, int k, float tol, int MAXITER)
{
    if(k > M->shape[1])
    {
        printf("ERROR:: File: %s, Line: %d, Function name: eigenvectors, ", __FILE__, __LINE__);
        printf("reason: %d > %d; The numbers of eigenvectors (k) must be at most equals to the rank of the matrix.\n", k, M->shape[1]);
        exit(EXIT_FAILURE); 
    }

    ndarray* A = cuda_ndarrayHost(M->shape[0], M->shape[1]);
    memcpy((void*)A->data, (void*)M->data,  M->shape[0] * M->shape[1] * sizeof(float));
  
    ndarray* E = cuda_ndarrayHost(k, A->shape[1]);
    ndarray* O = cuda_ndarrayHost(A->shape[1], 1);
    init(O, 0);

    float sqrterr = 0.f;

    for(int i = 0; i < k; i++)
    {
        int iter = 0;

        ndarray* x  = cuda_ndarrayHost(A->shape[1], 1);
        init(x, 1);

        ndarray* eigvc;
        do{
            eigvc = cudaMMProduct(A, x, false);
            
            float norm = cudaEDistance(eigvc, O, 1, false);
            cudaVSDivision(eigvc, norm, 1, true, false);

            sqrterr = cudaEDistance(eigvc, x, 1, false);
            cudaFreeHost_(x);

            x = eigvc;

        }while(sqrterr > tol && ++iter < MAXITER);

        memcpy((void*)&E->data[i * A->shape[1]], (void*)eigvc->data, eigvc->shape[0] * sizeof(float));

        ndarray* axeigvc = cudaMMProduct(A, eigvc, false);
        ndarray* eigvcT = cudaMTranspose(eigvc, 1, false);

        ndarray* eigva = cudaMMProduct(eigvcT, axeigvc, false);
        cudaFreeHost_(axeigvc);

        ndarray* m = cudaMMProduct(eigvc, eigvcT, false);
        cudaFreeHost_(eigvc);
        cudaFreeHost_(eigvcT);

        cudaMSProduct(m, eigva->data[0], 1, true, false);
        cudaFreeHost_(eigva);
        
        cudaMMSub(A, m, 1, true, false);
        cudaFreeHost_(m);

    }

    ndarray* ET = cudaMTranspose(E, 1, false);
    cudaFreeHost_(E);
    cudaFreeHost_(O);

    return ET;
}