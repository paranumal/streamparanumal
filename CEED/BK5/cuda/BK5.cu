/*

See LICENSE file.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "meshBasis.hpp"

void matrixPrint(int Nrows, int Ncols, dfloat *A, const char *mess){
#if 0
  printf("%s = [\n", mess);
  for(int i=0;i<Nrows;++i){
    for(int a=0;a<Ncols;++a){
      printf(" % e", A[i*Ncols+a]);
    }
    printf("\n");
  }
  printf("]\n");
#endif
}



__forceinline__ __device__ __host__  int ijN(const int i, const int j, const int N){

  return i + j*N;

}

__forceinline__ __device__ __host__ int ijkN(const int i, const int j, const int k, const int N){

  return i + j*N + k*N*N;

}

__forceinline__ __device__ __host__ int ijklN(const int i, const int j, const int k, const int l, const int N){

  return i + j*N + k*N*N + l*N*N*N;

}

// switch:
// 1 to use CUDA 10.0 stream recording
// 0 to use traditional enqueing of kernels
#define USE_GRAPH 0

#define MAX_DOFS_1D 14
#define MAX_HALF_DOFS_1D 7


#define HALF_DOFS_1D ((NUM_DOFS_1D+1)/2)

#define NUM_DOFS_2D (NUM_DOFS_1D*NUM_DOFS_1D)
#define NUM_DOFS_3D (NUM_DOFS_1D*NUM_DOFS_1D*NUM_DOFS_1D)

__constant__ dfloat const_DofToDofD[MAX_DOFS_1D*MAX_DOFS_1D];
__constant__ dfloat const_oddDofToDofD[MAX_HALF_DOFS_1D*MAX_HALF_DOFS_1D];
__constant__ dfloat const_evenDofToDofD[MAX_HALF_DOFS_1D*MAX_HALF_DOFS_1D];

void randAlloc(int N, dfloat **h_a, dfloat **c_a){

  *h_a = (dfloat*) calloc(N, sizeof(dfloat));

  for(int n=0;n<N;++n)
    h_a[0][n] = drand48();

  cudaMalloc(c_a, N*sizeof(dfloat));

  cudaMemcpy(c_a[0], h_a[0], N*sizeof(dfloat), cudaMemcpyHostToDevice);

}

__global__ void nothingKernel(){  }


template <int NUM_DOFS_1D, int p_Nblock >
  __forceinline__ __device__ 
  void BK5Device(const int numElements,
		 const int element,
		 const dfloat lambda,
		 const dfloat * __restrict__ op,
		 const dfloat * __restrict__ DofToDofD,
		 const dfloat * __restrict__ oddDofToDofD,
		 const dfloat * __restrict__ evenDofToDofD,
		 dfloat * __restrict__ r_p,
		 dfloat * __restrict__ r_Ap){
  
  __shared__ dfloat s_p[p_Nblock][NUM_DOFS_1D][NUM_DOFS_1D];
  __shared__ dfloat s_Gpr[p_Nblock][NUM_DOFS_1D][NUM_DOFS_1D];
  __shared__ dfloat s_Gps[p_Nblock][NUM_DOFS_1D][NUM_DOFS_1D];
  
  // assumes NUM_DOFS_2D threads
  int t = threadIdx.x;
  int blk = threadIdx.y;
  
  int i = t%NUM_DOFS_1D;
  int j = t/NUM_DOFS_1D;
  
  for(int k = 0; k < NUM_DOFS_1D; k++) {
    r_Ap[k] = 0.f; // zero the accumulator
  }
  
  // Layer by layer
#pragma unroll
  for(int k = 0; k < NUM_DOFS_1D; k++) {

    // share r_p[k]
    __syncthreads();

    s_p[blk][j][i] = r_p[k];

    __syncthreads();
    
    dfloat G00 = 0, G01 =0, G02 =0, G11 =0, G12 =0, G22 =0, GWJ =0;
    
    // prefetch geometric factors
    const int gbase = element*p_Nggeo*NUM_DOFS_3D + ijkN(i,j,k,NUM_DOFS_1D);

    if(element<numElements){
      G00 = op[gbase+p_G00ID*NUM_DOFS_3D];
      G01 = op[gbase+p_G01ID*NUM_DOFS_3D];
      G02 = op[gbase+p_G02ID*NUM_DOFS_3D];
      G11 = op[gbase+p_G11ID*NUM_DOFS_3D];
      G12 = op[gbase+p_G12ID*NUM_DOFS_3D];
      G22 = op[gbase+p_G22ID*NUM_DOFS_3D];
      GWJ = op[gbase+p_GWJID*NUM_DOFS_3D];
    }
    
    dfloat pr = 0.f;
    dfloat ps = 0.f;
    dfloat pt = 0.f;

#pragma unroll
    for(int m = 0; m < NUM_DOFS_1D; m++) {
      int im = ijN(m,i,NUM_DOFS_1D);
      int jm = ijN(m,j,NUM_DOFS_1D);
      int km = ijN(m,k,NUM_DOFS_1D);
      pr += DofToDofD[im]*s_p[blk][j][m];
      ps += DofToDofD[jm]*s_p[blk][m][i];
      pt += DofToDofD[km]*r_p[m];
    }
    
    s_Gpr[blk][j][i] = (G00*pr + G01*ps + G02*pt);
    s_Gps[blk][j][i] = (G01*pr + G11*ps + G12*pt);
    
    dfloat Gpt = (G02*pr + G12*ps + G22*pt);
    
    dfloat Apk = GWJ*lambda*r_p[k];
    
    __syncthreads();
    
#pragma unroll
    for(int m = 0; m < NUM_DOFS_1D; m++){
      int mi = ijN(i,m,NUM_DOFS_1D);
      int mj = ijN(j,m,NUM_DOFS_1D);
      int km = ijN(m,k,NUM_DOFS_1D);
      Apk     += DofToDofD[mi]*s_Gpr[blk][j][m];
      Apk     += DofToDofD[mj]*s_Gps[blk][m][i];
      r_Ap[m] += DofToDofD[km]*Gpt; // DT(m,k)*ut(i,j,k,e)
    }
    
    r_Ap[k] += Apk;
  }
  
}

template <int NUM_DOFS_1D, int p_Nblock >
__global__ void BK5ConstantKernel(const int numElements,
				  const dfloat lambda,
				  const dfloat * __restrict__ op,
				  const dfloat * __restrict__ DofToDofD,
				  const dfloat * __restrict__ oddDofToDofD,
				  const dfloat * __restrict__ evenDofToDofD,
				  const dfloat * __restrict__ solIn,
				  dfloat * __restrict__ solOut){
  
  __shared__ dfloat s_DofToDofD[NUM_DOFS_2D];

  dfloat r_q[NUM_DOFS_1D];
  dfloat r_Aq[NUM_DOFS_1D];

  const unsigned int t = threadIdx.x;
  const int blk = threadIdx.y;
  
  const int element = blockIdx.x*p_Nblock + blk;
  
  const unsigned int a = t%NUM_DOFS_1D;
  const unsigned int b = t/NUM_DOFS_1D;

  s_DofToDofD[t] = DofToDofD[t];
  
  if(element < numElements){
    for(int c=0;c<NUM_DOFS_1D;++c){
      
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      
      r_q[c] = solIn[id];
    }
  }
  
  __syncthreads();
  
  BK5Device  <NUM_DOFS_1D, p_Nblock>
    (numElements, element, lambda, op, s_DofToDofD, const_oddDofToDofD, const_evenDofToDofD, r_q, r_Aq);
  
  if(element<numElements){
#pragma unroll
    for(int c=0;c<NUM_DOFS_1D;++c){
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      solOut[id] = r_Aq[c];
    }
  }
}

template <int NUM_DOFS_1D>
  __forceinline__ __device__ 
  dfloat BK5CubeDevice(const int numElements,
		       const int element,
		       const dfloat lambda,
		       const dfloat * __restrict__ op,
		       const dfloat * __restrict__ DofToDofD,
		       dfloat r_p){
  
  __shared__ dfloat s_p[NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D];
  
  // assumes NUM_DOFS_2D threads
  int i = threadIdx.x;
  int j = threadIdx.y;
  int k = threadIdx.z;
  
  dfloat r_Ap = 0; // zero the accumulator

  s_p[k][j][i] = r_p;

  __syncthreads();
  
  dfloat G00 = 0, G01 =0, G02 =0, G11 =0, G12 =0, G22 =0, GWJ =0;
  
  // prefetch geometric factors
  const int gbase = element*p_Nggeo*NUM_DOFS_3D + ijkN(i,j,k,NUM_DOFS_1D);
  
  if(element<numElements){
    G00 = op[gbase+p_G00ID*NUM_DOFS_3D];
    G01 = op[gbase+p_G01ID*NUM_DOFS_3D];
    G02 = op[gbase+p_G02ID*NUM_DOFS_3D];
    G11 = op[gbase+p_G11ID*NUM_DOFS_3D];
    G12 = op[gbase+p_G12ID*NUM_DOFS_3D];
    G22 = op[gbase+p_G22ID*NUM_DOFS_3D];
    GWJ = op[gbase+p_GWJID*NUM_DOFS_3D];
  }

  r_Ap = GWJ*lambda*r_p;
  
  dfloat pr = 0.f;
  dfloat ps = 0.f;
  dfloat pt = 0.f;
  
#pragma unroll
  for(int m = 0; m < NUM_DOFS_1D; m++) {
    int im = ijN(m,i,NUM_DOFS_1D);
    int jm = ijN(m,j,NUM_DOFS_1D);
    int km = ijN(m,k,NUM_DOFS_1D);
    pr += DofToDofD[im]*s_p[k][j][m];
    ps += DofToDofD[jm]*s_p[k][m][i];
    pt += DofToDofD[km]*s_p[m][j][i];
  }
  
  dfloat Gpr = (G00*pr + G01*ps + G02*pt);
  dfloat Gps = (G01*pr + G11*ps + G12*pt);
  dfloat Gpt = (G02*pr + G12*ps + G22*pt);
  
  
  __syncthreads();

  s_p[k][j][i] = Gpr;

  __syncthreads();
  
#pragma unroll
  for(int m = 0; m < NUM_DOFS_1D; m++){
    int mi = ijN(i,m,NUM_DOFS_1D);
    r_Ap += DofToDofD[mi]*s_p[k][j][m];
  }


  __syncthreads();
  
  s_p[k][j][i] = Gps;

  __syncthreads();
  
#pragma unroll
  for(int m = 0; m < NUM_DOFS_1D; m++){
    int mj = ijN(j,m,NUM_DOFS_1D);
    r_Ap += DofToDofD[mj]*s_p[k][m][i];
  }

  __syncthreads();
  
  s_p[k][j][i] = Gpt;

  __syncthreads();
  
#pragma unroll
  for(int m = 0; m < NUM_DOFS_1D; m++){
    int mk= ijN(k,m,NUM_DOFS_1D);
    r_Ap += DofToDofD[mk]*s_p[m][j][i];
  }
  
  return r_Ap;
}

template <int NUM_DOFS_1D>
__global__ void BK5CubeKernel(const int numElements,
			       const dfloat lambda,
			       const dfloat * __restrict__ op,
			       const dfloat * __restrict__ DofToDofD,
			       const dfloat * __restrict__ solIn,
			       dfloat * __restrict__ solOut){
  
  __shared__ dfloat s_DofToDofD[NUM_DOFS_2D];
  
  const int element = blockIdx.x;
  
  int a = threadIdx.x;
  int b = threadIdx.y;
  int c = threadIdx.z;

  if(c==0)
    s_DofToDofD[b*NUM_DOFS_1D+a] = DofToDofD[b*NUM_DOFS_1D+a];
  
  int id = ijklN(a,b,c,element,NUM_DOFS_1D);
  
  dfloat r_p  = solIn[id];
  
  __syncthreads();
  
  dfloat r_Ap = BK5CubeDevice  <NUM_DOFS_1D>
    (numElements, element, lambda, op, s_DofToDofD, r_p);
  
  solOut[id] = r_Ap;

}



double bandwidthTest(cudaStream_t stream, int Ntests, size_t bwNtotal){

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);	
  
  dfloat *h_bwTest1, *c_bwTest1;
  dfloat *h_bwTest2, *c_bwTest2;
  
  randAlloc(bwNtotal/2, &h_bwTest1, &c_bwTest1);
  randAlloc(bwNtotal/2, &h_bwTest2, &c_bwTest2);
  
  cudaDeviceSynchronize();
  cudaEventRecord(start, stream);
  
  for(int test=0;test<Ntests/2;++test){
    cudaMemcpy(c_bwTest2, c_bwTest1, (bwNtotal/2)*sizeof(dfloat), cudaMemcpyDeviceToDevice);
    cudaMemcpy(c_bwTest1, c_bwTest2, (bwNtotal/2)*sizeof(dfloat), cudaMemcpyDeviceToDevice);
  }
  
  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);
  cudaDeviceSynchronize();

  float elapsed;
  cudaEventElapsedTime(&elapsed, start, end);
  elapsed /= 1000.; // convert to s
  elapsed /= (double) Ntests;
  
  double estimatedActualDeviceBandwidth = (bwNtotal*sizeof(dfloat)/elapsed)/1.e9;
  
  cudaFree(c_bwTest1);
  cudaFree(c_bwTest2);
  
  free(h_bwTest1);
  free(h_bwTest2);
  
  cudaEventDestroy(start);
  cudaEventDestroy(end);	
  
  return estimatedActualDeviceBandwidth;
}

// leave this here in case we add odd-even versions
void buildOddEvenMatrices(int NUM_COLS_OP, int NUM_ROWS_OP,
			  dfloat *h_OP,   dfloat **c_OP, dfloat **c_oddOP,  dfloat **c_evenOP){

  int HALF_COLS_OP = ((NUM_COLS_OP+1)/2);
  int HALF_ROWS_OP = ((NUM_ROWS_OP+1)/2);
  
  dfloat *X = (dfloat*) calloc(NUM_COLS_OP*NUM_COLS_OP, sizeof(dfloat));
  dfloat *invX = (dfloat*) calloc(NUM_COLS_OP*NUM_COLS_OP, sizeof(dfloat));

  dfloat *cubX = (dfloat*) calloc(NUM_ROWS_OP*NUM_ROWS_OP, sizeof(dfloat));
  dfloat *cubInvX = (dfloat*) calloc(NUM_ROWS_OP*NUM_ROWS_OP, sizeof(dfloat));

  for(int n=0;n<NUM_ROWS_OP;++n){
    cubX[n*NUM_ROWS_OP + n] = 1;
    cubInvX[n*NUM_ROWS_OP + n] = 0.5;

    if(n<NUM_ROWS_OP/2){
      cubX[n*NUM_ROWS_OP + NUM_ROWS_OP-1-n] = -1;
      cubInvX[n*NUM_ROWS_OP + NUM_ROWS_OP-1-n] = +0.5;
    }
    
    if(n>=(NUM_ROWS_OP/2)){
      cubX[n*NUM_ROWS_OP + NUM_ROWS_OP-1-n] = +1;
      cubInvX[n*NUM_ROWS_OP + NUM_ROWS_OP-1-n] = -0.5;
    }
  }

  for(int n=0;n<NUM_COLS_OP;++n){
    X[n*NUM_COLS_OP + n] = 1;
    invX[n*NUM_COLS_OP + n] = 0.5;

    if(n<NUM_COLS_OP/2){
      X[n*NUM_COLS_OP + NUM_COLS_OP-1-n] = 1;
      invX[n*NUM_COLS_OP + NUM_COLS_OP-1-n] = -0.5;
    }
    
    if(n>=NUM_COLS_OP/2){
      X[n*NUM_COLS_OP + NUM_COLS_OP-1-n] = -1;
      invX[n*NUM_COLS_OP + NUM_COLS_OP-1-n] = 0.5;
    }
  }
  
  if(NUM_COLS_OP%2) X[(NUM_COLS_OP)*(NUM_COLS_OP)/2] = 1;
  if(NUM_COLS_OP%2) invX[(NUM_COLS_OP)*(NUM_COLS_OP)/2] = 1;
  
  if(NUM_ROWS_OP%2) cubX[(NUM_ROWS_OP)*(NUM_ROWS_OP)/2] = 1;
  if(NUM_ROWS_OP%2) cubInvX[(NUM_ROWS_OP)*(NUM_ROWS_OP)/2] = 1;

  //  if(NUM_COLS_OP%2) invX[(NUM_COLS_OP)*(NUM_COLS_OP)/2] = 1;
  //  if(NUM_ROWS_OP%2) cubInvX[(NUM_ROWS_OP+1)*(NUM_ROWS_OP+1)/2] = 1;
  
  dfloat *IinvX = (dfloat*) calloc(NUM_COLS_OP*NUM_ROWS_OP, sizeof(dfloat));
  dfloat *cubInvXIinvX = (dfloat*) calloc(NUM_COLS_OP*NUM_ROWS_OP, sizeof(dfloat));

  // post multiply by invX
  for(int i=0;i<NUM_ROWS_OP;++i){
    for(int a=0;a<NUM_COLS_OP;++a){
      dfloat resI = 0;
      for(int n=0;n<NUM_COLS_OP;++n){
	resI += h_OP [i*NUM_COLS_OP+n]*invX[n*NUM_COLS_OP+a];
      }
      IinvX[i*NUM_COLS_OP+a] = resI;
    }
  }
  
  // pre multiply by invX
  for(int i=0;i<NUM_ROWS_OP;++i){
    for(int a=0;a<NUM_COLS_OP;++a){
      dfloat resI = 0;
      for(int n=0;n<NUM_ROWS_OP;++n){
	resI += cubInvX[i*NUM_ROWS_OP+n]*IinvX[n*NUM_COLS_OP + a];
      }
      cubInvXIinvX[i*NUM_COLS_OP+a] = resI;
    }
  }
  
  // now interleave the two non-zero blocks
  // [ A 0 ]  => [ A[0][0] B[0][0] A[0][1] B[0][1] .. A[0][HALF_DOFS_1D-1] B[0][HALF_DOFS_1D-1] .. 
  // [ 0 B ] 

  dfloat *oddOP  = (dfloat*) calloc(NUM_ROWS_OP*HALF_ROWS_OP, sizeof(dfloat));
  dfloat *evenOP = (dfloat*) calloc(NUM_ROWS_OP*HALF_ROWS_OP, sizeof(dfloat));
  
  for(int i=0;i<HALF_ROWS_OP;++i){
    for(int a=0;a<HALF_COLS_OP;++a){

      oddOP[i*HALF_COLS_OP+a]  = cubInvXIinvX[i*NUM_COLS_OP+a];
      evenOP[i*HALF_COLS_OP+a]  = cubInvXIinvX[(NUM_ROWS_OP-1-i)*NUM_COLS_OP + NUM_COLS_OP-1-a];
    }
  }

  if((NUM_ROWS_OP%2)) // zero duplicate
    evenOP[HALF_ROWS_OP*HALF_COLS_OP-1] = 0;
  
  int NoddOP  = HALF_ROWS_OP*HALF_COLS_OP;
  int NevenOP = HALF_ROWS_OP*HALF_COLS_OP;
  
  cudaMalloc(c_oddOP, NoddOP*sizeof(dfloat));
  cudaMalloc(c_evenOP, NevenOP*sizeof(dfloat));
  
  cudaMemcpy(*c_oddOP,  oddOP,  NoddOP*sizeof(dfloat),  cudaMemcpyHostToDevice);
  cudaMemcpy(*c_evenOP, evenOP, NoddOP*sizeof(dfloat), cudaMemcpyHostToDevice);

  cudaMemcpy(*c_OP, h_OP,  NUM_COLS_OP*NUM_ROWS_OP*sizeof(dfloat),  cudaMemcpyHostToDevice);

  matrixPrint(NUM_COLS_OP, NUM_COLS_OP, X, "X");
  matrixPrint(NUM_ROWS_OP, NUM_ROWS_OP, cubX, "cubX");

  
  matrixPrint(NUM_COLS_OP, NUM_COLS_OP, invX, "invX");
  matrixPrint(NUM_ROWS_OP, NUM_ROWS_OP, cubInvX, "cubInvX");


  
}


void runBK5Kernel(cudaStream_t stream, int Nq, int numElements, dfloat lambda,
		  dfloat *c_op,
		  dfloat *c_DofToDofD, dfloat *c_oddDofToDofD, dfloat *c_evenDofToDofD,
		  dfloat *c_solIn, dfloat *c_solOut, int mode){
  
#define BK5Kernel(Nq,Nblock)						\
  {									\
    if(mode==0){							\
      dim3 G((numElements+Nblock-1)/Nblock, 1, 1);			\
      dim3 B(Nq*Nq, Nblock, 1);						\
      BK5ConstantKernel<Nq,Nblock> <<< G, B, 0, stream >>>		\
	(numElements, lambda, c_op, c_DofToDofD, c_oddDofToDofD,c_evenDofToDofD, c_solIn, c_solOut); \
    }									\
    else{								\
      dim3 G(numElements,1,1);						\
      dim3 B(Nq, Nq, Nq);						\
      BK5CubeKernel<Nq> <<< G, B, 0, stream >>>				\
	(numElements, lambda, c_op, c_DofToDofD, c_solIn, c_solOut);	\
    }									\
}
  
  
#define ERR printf("massMatrixMultiplyRegister with Nq=%d not available", Nq); exit(-1)
  
  if(Nq==2){
    BK5Kernel(2,16);
    return;
  }
  
  if(Nq==3){
    BK5Kernel(3,7);
    return;
  }

  if(Nq==4){
    BK5Kernel(4,4);
    return;
  }

  if(Nq==5){
    BK5Kernel(5,5);
    return;
  }

  if(Nq==6){
    BK5Kernel(6,3);
    return;
  }

  if(Nq==7){
    BK5Kernel(7,2);
    return;
  }

  if(Nq==8){
    BK5Kernel(8,1);
    return;
  }

  if(Nq==9){
    BK5Kernel(9,1);
    return;
  }

  if(Nq==10){
    BK5Kernel(10,1);
    return;
  }

  if(Nq==11){
    BK5Kernel(11,1);
    return;
  }
  
  if(Nq==12){
    BK5Kernel(12,1);
    return;
  }

  if(Nq==13){
    BK5Kernel(13,1);
    return;
  }

  ERR;
}


dfloat nothingTest(cudaStream_t stream, int Ntests){

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);	

  cudaDeviceSynchronize();
  
  float nothingElapsed = 0;
  {
    
    // time kernel that does nothing
    
#if USE_GRAPH==1
    // cuda stream capture sequence for nothingKernel
    cudaGraph_t nothingGraph;
    
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    for(int test=0;test<Ntests;++test){
      nothingKernel <<< 1, 1, 0, stream >>> ();
    }
    
    cudaStreamEndCapture(stream, &nothingGraph);
    
    // time graph sequence for nothing
    cudaGraphExec_t nothingInstance;
    cudaGraphInstantiate(&nothingInstance, nothingGraph, NULL, NULL, 0);
    
    cudaEventRecord(start, stream);
    
    cudaGraphLaunch(nothingInstance, stream);
    
    cudaEventRecord(end, stream);
#else
    
    cudaEventRecord(start, stream);
    
    for(int test=0;test<Ntests;++test)
      nothingKernel <<< 1, 1, 0, stream >>> ();
    
    cudaEventRecord(end, stream);
    
#endif
    
    cudaDeviceSynchronize();
    
    cudaEventElapsedTime(&nothingElapsed, start, end);
    nothingElapsed /= 1000.;
    nothingElapsed /= (double) Ntests;
    
  }

  return nothingElapsed;
}


int main(int argc, char **argv){

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  if(argc!=4){
    printf("Usage: ./BK5 Nq numElements mode\n");
    exit(-1);
  }

  // read number of elements
  int          Nq = atoi(argv[1]);
  int numElements = atoi(argv[2]);
  int        mode = atoi(argv[3]);
  
  dfloat lambda = 0;
  
  printf("Running: NUM_DOFS_1D=%d, numElements=%d\n", Nq, numElements);

  int   Np = Nq*Nq*Nq;
  int halfNq = ((Nq+1)/2);

  int    Ntotal = numElements*Np;

  int Ntests = 10;
  
  double estimatedActualDeviceBandwidth = bandwidthTest(stream, Ntests, (Ntotal*2+7*Ntotal)*sizeof(dfloat));
  
  dfloat *h_op,      *c_op;
  dfloat *h_solOut,       *c_solOut;
  dfloat *h_solIn,        *c_solIn;

  dfloat *h_DofToDofD,    *c_DofToDofD;
  dfloat *c_oddDofToDofD, *c_evenDofToDofD;

  // float fields
  randAlloc(Ntotal*p_Nggeo, &h_op, &c_op);
  
  randAlloc(Ntotal, &h_solIn, &c_solIn);
  randAlloc(Ntotal, &h_solOut, &c_solOut);
  
  randAlloc(Nq*Nq, &h_DofToDofD, &c_DofToDofD);
  
  // give D the correct symmetry
  for(int i=0;i<halfNq;++i){
    for(int a=0;a<Nq;++a){
      h_DofToDofD[(Nq-1-i)*Nq + Nq-1-a] = -h_DofToDofD[i*Nq+a];
    }
  }

  // create Odd-even packed storage for I and transpose(I) and push to constant memory
  buildOddEvenMatrices (Nq,Nq, h_DofToDofD, &c_DofToDofD, &c_oddDofToDofD, &c_evenDofToDofD);

  cudaMemcpyToSymbol(const_DofToDofD,     c_DofToDofD,     Nq*Nq*sizeof(dfloat), 0, cudaMemcpyDeviceToDevice);
  cudaMemcpyToSymbol(const_oddDofToDofD,  c_oddDofToDofD,  halfNq*halfNq*sizeof(dfloat), 0, cudaMemcpyDeviceToDevice);
  cudaMemcpyToSymbol(const_evenDofToDofD, c_evenDofToDofD, halfNq*halfNq*sizeof(dfloat), 0, cudaMemcpyDeviceToDevice);
  
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);	

  // KERNEL GRID
  // do nothing kernel test
  dfloat nothingElapsed = nothingTest(stream, Ntests);
  nothingElapsed = nothingTest(stream, Ntests);
  
  // warm up call
  runBK5Kernel (stream, Nq, numElements, lambda,
		c_op,
		c_DofToDofD, c_oddDofToDofD, c_evenDofToDofD,
		c_solIn, c_solOut, mode);
  
#if USE_GRAPH==1
  // cuda stream capture
  cudaGraph_t graph;
  
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  for(int test=0;test<Ntests;++test){

    runMassBK5Kernel (stream, Nq, numElements, lambda,
		      c_op,
		      c_DofToDofD, c_oddDofToDofD, c_evenDofToDofD,
		      c_solIn, c_solOut, mode);
  }
  
  cudaStreamEndCapture(stream, &graph);
  
  cudaGraphExec_t instance;
  cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
#endif
  
  cudaDeviceSynchronize();

  {
    cudaEventRecord(start, stream);
    
#if USE_GRAPH==0
    for(int test=0;test<Ntests;++test){

      runBK5Kernel (stream, Nq, numElements, lambda,
		    c_op,
		    c_DofToDofD, c_oddDofToDofD, c_evenDofToDofD,
		    c_solIn, c_solOut, mode);
      
    }
#else
    cudaGraphLaunch(instance, stream);
#endif

    cudaEventRecord(end, stream);
    
    cudaEventSynchronize(end);
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, end);
    elapsed /= 1000.;
    elapsed /= (double) Ntests;

    int bytesMoved = (2*Np+7*Np)*sizeof(dfloat); // x, Mx, opa   
    double bw = (bytesMoved*numElements/elapsed)/1.e9;

    double flopCount = Np*(6*2*Nq + 17);
    double gflops = (flopCount*numElements/elapsed)/1.e9;
    
    printf("%2d %8d %8d %e %e %e %e %e %e %%%% [BK5: N, numElements, Ndofs,"
	   " elapsed, dofsPerSecond, nothingElapsed, BW in GB/s, estimatedActualDeviceBandwidth, GFLOPS/s]\n",
	   Nq-1, numElements, Np*numElements, elapsed, numElements*(Np/elapsed),
	   nothingElapsed, bw, estimatedActualDeviceBandwidth, gflops);
  }

  // check output is correct
  //  BK5Host (Nq, numElements, lambda, h_op, h_DofToDofD, h_solIn, h_solOut);
  meshReferenceBK5(Nq, numElements, lambda, h_op, h_DofToDofD, h_solIn, h_solOut);

  // copy device version to host old qh
  dfloat *fromDevice = (dfloat*) calloc(numElements*Np, sizeof(dfloat));
  cudaMemcpy(fromDevice, c_solOut, numElements*Np*sizeof(dfloat), cudaMemcpyDeviceToHost);

  dfloat maxDiff = 0;
  
  for(int e=0;e<numElements;++e){
    for(int n=0;n<Np;++n){
      int id = e*Np + n;
      dfloat diff = fabs(h_solOut[id]-fromDevice[id]);
      maxDiff = (diff>maxDiff) ? diff:maxDiff;
    }
  }
  printf("|| Mq_{host} - Mq_{device} ||_linf = %lg\n", maxDiff);
  
  cudaEventDestroy(start);
  cudaEventDestroy(end);	
  
  return 0;

}
