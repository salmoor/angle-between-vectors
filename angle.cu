/**
 * Alemdar Salmoor
 * */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>

//The following implementation of the atomicAdd for devices with compute capabilities lower
//than 6.0 is provided on the NVidia Cuda Toolkit Documentation page.

__device__ double myAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

void arrayGenerator(int * arr, int N);
double dotProduct(int * A, int * B, int N);
double serial(int * A, int * B, int N);
double parallel(int * A, int * B, int N, int, int *, double *, double *, double *, double *);
void output(int, int, int, double, double, double, double, double, double, double, double);
double computeTime(clock_t dif);
int myMin(int, int);
void computeBlocksAndSmallestArray(int, int, int, int, int, int *, int *, int *);
int getMaxBlocks(int, int);
void myPrintDash(int range);
void readArrays(char *, int ** , int **, int *);
double firstLargerN(int target);
int computeExtraBlocks(int N, int base, int blockSize, int intsPerThread);
__global__ void angleKernel(int * A, int * B, int N, double * dab, double * daa, double * dbb, int);




int main(int argc, char ** argv){

    //Seed
    srand(time(NULL));

    //Time
    clock_t mTime;

    int N = atoi(argv[1]);
    int blockSize = atoi(argv[2]);
    char * input;

    //Output
    int elems, tpb, blocks;
    double tAG, tCPU, tHDT, tKernel, tDHT, tGPU;
    double rCPU, rGPU;

    elems = N;
    tpb = blockSize;

    int * A; 
    int * B;

    //printf("Argc: %d\n", argc);


    mTime = clock();
    if(argc == 4){
        input = argv[3];
        readArrays(input, &A, &B, &N);
    }
    else{
        A = (int *) malloc(N * sizeof(int));
        B = (int *) malloc(N * sizeof(int));

        arrayGenerator(A, N);
        arrayGenerator(B, N);
    }
    mTime = clock() - mTime;
    tAG = computeTime(mTime);

    mTime = clock();
    rCPU = serial(A, B, N);
    mTime = clock() - mTime;
    tCPU = computeTime(mTime);

    rGPU = parallel(A, B, N, blockSize, &blocks, &tHDT, &tKernel, &tDHT, &tGPU);



    free(A);
    free(B);

    output(elems, tpb, blocks, tAG, tCPU, tHDT, tKernel, tDHT, tGPU, rCPU, rGPU);

    return 0;
}


void arrayGenerator(int *arr, int N){

    for (size_t i = 0; i < N; i++)
    {
        arr[i] = rand();
        arr[i] = arr[i] - RAND_MAX/2;
    }
    
}

double serial(int * A, int * B, int N){

    double numerator, denominator, A_squared, B_squared;
    double cos_angle, angle;


    numerator = dotProduct(A, B, N);

    A_squared = dotProduct(A, A, N);
    B_squared = dotProduct(B, B, N);

    denominator = A_squared * B_squared;
    denominator = sqrt(denominator);

    cos_angle = numerator/denominator;

    angle = acos(cos_angle);
    angle = angle * (180.0/M_PI);
    
    return angle;

}

double dotProduct(int * A, int * B, int N){
    
    double product = 0.0;

    for (size_t i = 0; i < N; i++){
        product += (double) A[i] * (double) B[i];
    }

    return product;
    
}

void output(int elems, int tpb, int blocks, double tAG, double tCPU, double tHDT, double tKernel, double tDHT, double tGPU, double rCPU, double rGPU){

    printf("\n");
    printf("Info\n");
    myPrintDash(14);
    printf("Number of elements: %d\n", elems);
    printf("Number of threads per block: %d\n", tpb);
    printf("Number of blocks will be created: %d\n", blocks);
    printf("\n");
    printf("Time\n");
    myPrintDash(14);
    printf("Time for the array generation: %lf ms\n", tAG);
    printf("Time for the CPU function: %lf ms\n", tCPU);
    printf("Time for the Host to Device transfer: %lf ms\n", tHDT);
    printf("Time for the kernel execution: %lf ms\n", tKernel);
    printf("Time for the Device to Host transfer: %lf ms\n", tDHT);
    printf("Total execution time for GPU: %lf ms\n", tGPU);
    printf("\n");
    printf("Results\n");
    myPrintDash(14);
    printf("CPU result: %.3lf\n", rCPU);
    printf("GPU result: %.3lf\n", rGPU);
    printf("\n");

}

double computeTime(clock_t dif){
    
    double e = (double) dif;
    e = e/CLOCKS_PER_SEC;
    e = e * 1000.0;

    return e;
}

double parallel(int * A, int * B, int N, int blockSize, int * blocksCreated, double * tHDT, double * tKernel, double * tDHT, double * tGPU){

    clock_t mTime;

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    int multiProcs = p.multiProcessorCount;
    int maxThreads = p.maxThreadsPerMultiProcessor;
    int major = p.major;
    int minor = p.minor;
    int maxBlocks = getMaxBlocks(major, minor);
    int smallestArraySize, numBlocks, INTS_PER_THREAD;

    computeBlocksAndSmallestArray(multiProcs, blockSize, maxBlocks, maxThreads, N, &smallestArraySize, &numBlocks, &INTS_PER_THREAD);

    //output variables
    double numerator, A_squared, B_squared;


    //device arrays;
    int * D_A;
    int * D_B;
    double * dotAB;
    double * dotAA;
    double * dotBB;

    

    //Allocate for device
    cudaMalloc(&D_A, smallestArraySize * sizeof(int));
    cudaMalloc(&D_B, smallestArraySize * sizeof(int));
    cudaMalloc(&dotAB, sizeof(double));
    cudaMalloc(&dotAA, sizeof(double));
    cudaMalloc(&dotBB, sizeof(double));

    size_t size = N * sizeof(int);

    //copy from host to device
    mTime = clock();
    cudaMemcpy(D_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(D_B, B, size, cudaMemcpyHostToDevice);
    mTime = clock() - mTime;
    (*tHDT) = computeTime(mTime);

    int surplus = (smallestArraySize - N) * sizeof(int);

    cudaMemset(dotAA, 0, sizeof(double));
    cudaMemset(dotAB, 0, sizeof(double));
    cudaMemset(dotBB, 0, sizeof(double));
    cudaMemset((D_A + N), 0, surplus);
    cudaMemset((D_B + N), 0, surplus);
    

    int sharedMem = 3 * sizeof(double) * blockSize;

    mTime = clock();
    angleKernel<<<numBlocks, blockSize, sharedMem>>>(D_A, D_B, N, dotAB, dotAA, dotBB, INTS_PER_THREAD);
    cudaDeviceSynchronize();
    mTime = clock() - mTime;
    (*tKernel) = computeTime(mTime);

    //copy from device to host
    mTime = clock();
    cudaMemcpy(&numerator, dotAB, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&A_squared, dotAA, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&B_squared, dotBB, sizeof(double), cudaMemcpyDeviceToHost);
    mTime = clock() - mTime;
    (*tDHT) = computeTime(mTime);


    double denominator = A_squared * B_squared;
    denominator = sqrt(denominator);

    double cos_angle = numerator/denominator;

    double angle = acos(cos_angle);
    angle = angle * (180.0/M_PI);

    (*tGPU) = (*tHDT) + (*tKernel) + (*tDHT);
    (*blocksCreated) = numBlocks;

    cudaFree(D_A);
    cudaFree(D_B);
    cudaFree(dotAB);
    cudaFree(dotAA);
    cudaFree(dotBB);

    return angle;
}



__global__ void angleKernel(int * A, int * B, int N, double * dotAB, double * dotAA, double * dotBB, int intsPerThread){

    extern __shared__ double sharedMem[];

    int gid = threadIdx.x * intsPerThread + blockIdx.x * blockDim.x * intsPerThread;
    int tid = threadIdx.x;
    int gend = gid + intsPerThread;

    double * localAB = sharedMem;
    double * localAA = (localAB + blockDim.x);
    double * localBB = (localAA + blockDim.x);
    
    double Aval, Bval;

    localAB[tid] = 0.0; localAA[tid] = 0.0; localBB[tid] = 0.0;

    
    for(int i = gid; i < gend; i++){
        Aval = (double) A[i];
        Bval = (double) B[i];

        localAB[tid] += Aval * Bval;
        localAA[tid] += Aval * Aval;
        localBB[tid] += Bval * Bval;

    }

    int size = blockDim.x / 2;

    while(size > 0){
        __syncthreads();

        if(tid < size){
            localAB[tid] += localAB[tid + size];
            localAA[tid] += localAA[tid + size];
            localBB[tid] += localBB[tid + size];
        }

        size = size/2;

    }

    if(tid == 0){
        myAtomicAdd(dotAB, localAB[tid]);
        myAtomicAdd(dotAA, localAA[tid]);
        myAtomicAdd(dotBB, localBB[tid]);
    }

}


void computeBlocksAndSmallestArray(int multiProcs, int blockSize, int maxBlocks, int maxThreads, int N, int * smallestArraySize, int * numBlocks, int * INTS_PER_THREAD){

    int maxThreadChunk = maxThreads/blockSize;

    int activeBlocks = myMin(maxBlocks, maxThreadChunk);

    int base = multiProcs * activeBlocks;

    double largerN = firstLargerN(N);

    double logN = log2(largerN);

    (* INTS_PER_THREAD) =  (int) ceil(logN);

    int extraBlocks = computeExtraBlocks(N, base, blockSize, (* INTS_PER_THREAD));

    (* numBlocks) = base + extraBlocks;

    (* smallestArraySize) = (* numBlocks) * blockSize * (* INTS_PER_THREAD);

    // printf("MultiProcs: %d\n", multiProcs);
    // printf("maxBlocks: %d\n", maxBlocks);
    // printf("maxThreads: %d\n", maxThreads);
    // printf("MaxThreadChunk: %d\n", maxThreadChunk);
    // printf("ActiveBlocks: %d\n", activeBlocks);
    // printf("base: %d\n", base);
    // printf("LargerN: %lf\n", largerN);
    // printf("LogN: %lf\n", logN);
    // printf("IntsPerThread: %d\n", (* INTS_PER_THREAD));
    // printf("Extra Blocks: %d\n", extraBlocks);
    // printf("NumBlocks: %d\n", (*numBlocks));
    // printf("smallestArraySize: %d\n", (*smallestArraySize));
}

int getMaxBlocks(int major, int minor){

    int maxBlocks = 16;

    if(major == 5 || major == 6 || (major == 7 && minor == 0)){
        maxBlocks = 32;
    }

    return maxBlocks;

}

int myMin(int A, int B){

    if(A < B) return A;
    else return B;
}

void myPrintDash(int range){

    for(int i = 0; i < range; i++){
        printf("\u2012");
    }
    printf("\n");
}

void readArrays(char * input, int ** A, int ** B, int * N){

    FILE * finput = fopen(input, "r");

    fscanf(finput, "%d\n", N);
    int num, i;

    (* A) = (int *) malloc((* N) * sizeof(int));
    (* B) = (int *) malloc((* N) * sizeof(int));


    for(i = 0; i < (* N); i++){
        fscanf(finput, "%d\n", &num);
        (* A)[i] = num;
    }

    for(i = 0; i < (* N); i++){
        fscanf(finput, "%d\n", &num);
        (* B)[i] = num;
    }

    fclose(finput);

}

double firstLargerN(int t){

    double target = (double) t;

    double accumulator = 1;

    while(accumulator < target){
        accumulator *= 2;
    }

    return accumulator;

}

int computeExtraBlocks(int N, int base, int blockSize, int intsPerThread){

    double dN = (double) N;
    double dBase = (double) base;
    double dBlockSize = (double) blockSize;
    double ipt = (double) intsPerThread;
    
    double result = dN/(dBlockSize * ipt) - dBase;

    return (int) ceil(result);
    
}

// printf("Rand Max: %d\n", RAND_MAX);
// printf("Int Max: %d\n", INT_MAX);
// printf("Double Max: %lf\n", DBL_MAX);


// printf("sizeof int: %lu\n", sizeof(int));
// printf("Shared mem per block: %lu\n", p.sharedMemPerBlock);
// printf("Warp size: %d\n", p.warpSize);
// printf("Multiprocessor count: %d\n", p.multiProcessorCount);
// printf("Major: %d\n", p.major);
// printf("Minor: %d\n", p.minor);
// printf("Max Threads per Multiprocessor: %d\n", p.maxThreadsPerMultiProcessor);


// if(blockIdx.x == gridDim.x - 1){
//     printf("my tid: %d, gid: %d, and gend: %d\n", tid, gid, gend);
// }


// if(gid == 0){
//     printf("Parallel: ");
//     for(int i = 0; i < gridDim.x * blockDim.x * intsPerThread; i++){
//         printf("%d ", A[i]);
//     }
//     printf("\n");
// }

// if(tid == 0){
//     printf("My blockId: %d, localAB: %d\n", blockIdx.x, localAB[tid]);
// }


// if(argc == 4){
//     printf("Input file: %s\n", input);
//     printf("N: %d\n", N);

//     printf("A: ");
//     for(int i = 0; i < N; i++){
//         printf("%d, ", A[i]);
//     }
//     printf("\n");

//     printf("B: ");
//     for(int i = 0; i < N; i++){
//         printf("%d, ", B[i]);
//     }
//     printf("\n");

// }

//printf("Vector size: %d\n", N);
//printf("Block Size: %d\n", blockSize);