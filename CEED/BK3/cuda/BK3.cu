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

#define MAX_QUAD_1D 16
#define MAX_DOFS_1D 14
#define MAX_HALF_QUAD_1D 8
#define MAX_HALF_DOFS_1D 7


#define HALF_DOFS_1D ((NUM_DOFS_1D+1)/2)
#define HALF_QUAD_1D ((NUM_QUAD_1D+1)/2)

#define p_padCubNq  0
//((NUM_QUAD_1D%4) ? 0:1)

#define NUM_DOFS_2D (NUM_DOFS_1D*NUM_DOFS_1D)
#define NUM_DOFS_3D (NUM_DOFS_1D*NUM_DOFS_1D*NUM_DOFS_1D)

#define NUM_QUAD_2D (NUM_QUAD_1D*NUM_QUAD_1D)
#define NUM_QUAD_3D (NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D)

__constant__ dfloat const_DofToQuad[MAX_DOFS_1D*MAX_QUAD_1D];
__constant__ dfloat const_oddDofToQuad[MAX_HALF_QUAD_1D*MAX_HALF_DOFS_1D];
__constant__ dfloat const_evenDofToQuad[MAX_HALF_QUAD_1D*MAX_HALF_DOFS_1D];

__constant__ dfloat const_QuadToQuadD[MAX_QUAD_1D*MAX_QUAD_1D];
__constant__ dfloat const_oddQuadToQuadD[MAX_HALF_QUAD_1D*MAX_HALF_QUAD_1D];
__constant__ dfloat const_evenQuadToQuadD[MAX_HALF_QUAD_1D*MAX_HALF_QUAD_1D];

void randAlloc(int N, dfloat **h_a, dfloat **c_a){

  *h_a = (dfloat*) calloc(N, sizeof(dfloat));

  for(int n=0;n<N;++n)
    h_a[0][n] = drand48();

  cudaMalloc(c_a, N*sizeof(dfloat));

  cudaMemcpy(c_a[0], h_a[0], N*sizeof(dfloat), cudaMemcpyHostToDevice);

}

__global__ void nothingKernel(){  }


template <int NUM_DOFS_1D, int NUM_QUAD_1D, int p_Nblock >
  __forceinline__ __device__ 
  void BK3InnerDevice(const int numElements,
		      const int element,
		      const dfloat lambda,
		      const dfloat * __restrict__ op,
		      const dfloat * __restrict__ QuadToQuadD,
		      const dfloat * __restrict__ oddQuadToQuadD,
		      const dfloat * __restrict__ evenQuadToQuadD,
		      dfloat s_Ap[p_Nblock][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D+p_padCubNq],
		      dfloat * __restrict__ r_Ap){

  __shared__ dfloat s_Gpr[p_Nblock][NUM_QUAD_1D][NUM_QUAD_1D];
  __shared__ dfloat s_Gps[p_Nblock][NUM_QUAD_1D][NUM_QUAD_1D];
  
  dfloat r_p[NUM_QUAD_1D];
  
  // assumes NUM_QUAD_2D threads
  int t = threadIdx.x;
  int blk = threadIdx.y;
  
  int i = t%NUM_QUAD_1D;
  int j = t/NUM_QUAD_1D;
  
  for(int k = 0; k < NUM_QUAD_1D; k++) {
    r_p[k] = s_Ap[blk][k][j][i];; // prefetch operation
    r_Ap[k] = 0.f; // zero the accumulator
  }
  
  // Layer by layer
#pragma unroll
  for(int k = 0; k < NUM_QUAD_1D; k++) {

    dfloat G00 = 0, G01 =0, G02 =0, G11 =0, G12 =0, G22 =0, GWJ =0;
    
    // prefetch geometric factors
    const int gbase = element*p_Nggeo*NUM_QUAD_3D + ijkN(i,j,k,NUM_QUAD_1D);

    if(element<numElements){
      G00 = op[gbase+p_G00ID*NUM_QUAD_3D];
      G01 = op[gbase+p_G01ID*NUM_QUAD_3D];
      G02 = op[gbase+p_G02ID*NUM_QUAD_3D];
      G11 = op[gbase+p_G11ID*NUM_QUAD_3D];
      G12 = op[gbase+p_G12ID*NUM_QUAD_3D];
      G22 = op[gbase+p_G22ID*NUM_QUAD_3D];
      GWJ = op[gbase+p_GWJID*NUM_QUAD_3D];
    }
    
    dfloat pr = 0.f;
    dfloat ps = 0.f;
    dfloat pt = 0.f;

#pragma unroll
    for(int m = 0; m < NUM_QUAD_1D; m++) {
      int im = ijN(m,i,NUM_QUAD_1D);
      int jm = ijN(m,j,NUM_QUAD_1D);
      int km = ijN(m,k,NUM_QUAD_1D);
      pr += QuadToQuadD[im]*s_Ap[blk][k][j][m];
      ps += QuadToQuadD[jm]*s_Ap[blk][k][m][i];
      pt += QuadToQuadD[km]*r_p[m];
    }

    __syncthreads();
    
    s_Gpr[blk][j][i] = (G00*pr + G01*ps + G02*pt);
    s_Gps[blk][j][i] = (G01*pr + G11*ps + G12*pt);
    
    dfloat Gpt = (G02*pr + G12*ps + G22*pt);
    
    dfloat Apk = GWJ*lambda*r_p[k];
    
    __syncthreads();
    
#pragma unroll
    for(int m = 0; m < NUM_QUAD_1D; m++){
      int mi = ijN(i,m,NUM_QUAD_1D);
      int mj = ijN(j,m,NUM_QUAD_1D);
      int km = ijN(m,k,NUM_QUAD_1D);
      Apk     += QuadToQuadD[mi]*s_Gpr[blk][j][m];
      Apk     += QuadToQuadD[mj]*s_Gps[blk][m][i];
      r_Ap[m] += QuadToQuadD[km]*Gpt; // DT(m,k)*ut(i,j,k,e)
    }
    
    r_Ap[k] += Apk;
  }
  
  __syncthreads();
  
  for(int k=0;k<NUM_QUAD_1D;++k){
    s_Ap[blk][k][j][i] = r_Ap[k];

  }
  
  __syncthreads();
  
}

template <int NUM_DOFS_1D, int NUM_QUAD_1D, int p_Nblock >
  __forceinline__ __device__ 
  void BK3OuterDevice(const int numElements,
		      const int element,
		      const dfloat lambda,
		      const dfloat * __restrict__ op,
		      const dfloat * __restrict__ oddDofToQuad,
		      const dfloat * __restrict__ evenDofToQuad,
		      const dfloat * __restrict__ QuadToQuadD,
		      const dfloat * __restrict__ oddQuadToQuadD,
		      const dfloat * __restrict__ evenQuadToQuadD,
		      dfloat s_Ap[p_Nblock][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D+p_padCubNq],
		      dfloat * __restrict__ r_Ap){
  
  dfloat r_tmpOdd[HALF_QUAD_1D];
  dfloat r_tmpEven[HALF_QUAD_1D];

  const int t   = threadIdx.x;
  const int blk = threadIdx.y;
  
  // assumes barrier before s_Ap was used last
  
  // transform in 'c'
  if(t<NUM_DOFS_2D){

    const int a = t%NUM_DOFS_1D;
    const int b = t/NUM_DOFS_1D;
    
#pragma unroll
    for(int c=0;c<HALF_DOFS_1D;++c){
      r_tmpOdd[c]  = r_Ap[c] + r_Ap[NUM_DOFS_1D-1-c];
      r_tmpEven[c] = r_Ap[c] - r_Ap[NUM_DOFS_1D-1-c];
    }

    if(NUM_DOFS_1D%2)
      r_tmpOdd[HALF_DOFS_1D-1] *= 0.5f;
    
#pragma unroll
    for(int k=0;k<HALF_QUAD_1D;++k){
      dfloat resOdd = 0, resEven = 0;
      
#pragma unroll
      for(int c=0;c<HALF_DOFS_1D;++c){
	int kc = ijN(c,k,HALF_DOFS_1D);		
	resOdd  += oddDofToQuad[kc]*r_tmpOdd[c];
	resEven += evenDofToQuad[kc]*r_tmpEven[c];
      }
      
      s_Ap[blk][k][b][a]               = resOdd + resEven;
      s_Ap[blk][NUM_QUAD_1D-1-k][b][a] = resOdd - resEven;
    }
    
  }
  
  __syncthreads();

  // transform in 'b'
  if(t<NUM_DOFS_1D*NUM_QUAD_1D){
    const int a = t%NUM_DOFS_1D;
    const int k = t/NUM_DOFS_1D;
    
#pragma unroll
    for(int b=0;b<HALF_DOFS_1D;++b){
      dfloat ApOdd  = s_Ap[blk][k][b][a];
      dfloat ApEven = s_Ap[blk][k][NUM_DOFS_1D-1-b][a];
      r_tmpOdd[b]  = ApOdd + ApEven;
      r_tmpEven[b] = ApOdd - ApEven;
    }      
    
    if(NUM_DOFS_1D%2)
      r_tmpOdd[HALF_DOFS_1D-1] *= 0.5f;
    
#pragma unroll
    for(int j=0;j<HALF_QUAD_1D;++j){
      dfloat resOdd = 0, resEven = 0;
      
#pragma unroll
      for(int b=0;b<HALF_DOFS_1D;++b){
	int jb = ijN(b,j,HALF_DOFS_1D);
	resOdd  += oddDofToQuad[jb]*r_tmpOdd[b];
	resEven += evenDofToQuad[jb]*r_tmpEven[b];
      }
      
      s_Ap[blk][k][j][a]               = resOdd+resEven;
      s_Ap[blk][k][NUM_QUAD_1D-1-j][a] = resOdd-resEven;
      
    }
  }
  
  __syncthreads();

  // transform in 'a'
  {
    const int j = t%NUM_QUAD_1D;
    const int k = t/NUM_QUAD_1D;
    
#pragma unroll
    for(int a=0;a<HALF_DOFS_1D;++a){
      dfloat ApOdd  = s_Ap[blk][k][j][a];
      dfloat ApEven = s_Ap[blk][k][j][NUM_DOFS_1D-1-a];
      r_tmpOdd[a]  = ApOdd + ApEven;
      r_tmpEven[a] = ApOdd - ApEven;
    }

    if(NUM_DOFS_1D%2)
      r_tmpOdd[HALF_DOFS_1D-1] *= 0.5f;

    
#pragma unroll
    for(int i=0;i<HALF_QUAD_1D;++i){
      dfloat resOdd = 0, resEven = 0;
      
#pragma unroll
      for(int a=0;a<HALF_DOFS_1D;++a){
	int ia = ijN(a,i,HALF_DOFS_1D);
	resOdd  += oddDofToQuad[ia]*r_tmpOdd[a];
	resEven += evenDofToQuad[ia]*r_tmpEven[a];
      }

      
      s_Ap[blk][k][j][i] = resOdd + resEven;
      s_Ap[blk][k][j][NUM_QUAD_1D-1-i] = resOdd - resEven;
    }
  }
  
  __syncthreads();

  // enters in s_Ap, leaves in r_Ap
  BK3InnerDevice <NUM_DOFS_1D, NUM_QUAD_1D, p_Nblock> (numElements, element, lambda, op,
						       QuadToQuadD, oddQuadToQuadD, evenQuadToQuadD, s_Ap, r_Ap);
  
  __syncthreads();
  
  // test in 'a'
  {
    const int j = t%NUM_QUAD_1D;
    const int k = t/NUM_QUAD_1D;
    
    // need to load from s_Ap into r_Ap
    
#pragma unroll
    for(int i=0;i<HALF_QUAD_1D;++i){
      dfloat ApOdd  = s_Ap[blk][k][j][i];
      dfloat ApEven = s_Ap[blk][k][j][NUM_QUAD_1D-1-i];
      r_tmpOdd[i]  = ApOdd + ApEven;
      r_tmpEven[i] = ApOdd - ApEven;
    }      

    if(NUM_QUAD_1D%2)
      r_tmpOdd[HALF_QUAD_1D-1] *= 0.5f;

    
#pragma unroll
    for(int a=0;a<HALF_DOFS_1D;++a){
      dfloat resOdd = 0, resEven = 0;
      
#pragma unroll
      for(int i=0;i<HALF_QUAD_1D;++i){
	int ia = ijN(a,i,HALF_DOFS_1D);
	resOdd  += oddDofToQuad[ia]*r_tmpOdd[i];
	resEven += evenDofToQuad[ia]*r_tmpEven[i];
      }
      
      s_Ap[blk][k][j][a]               = resOdd + resEven;
      s_Ap[blk][k][j][NUM_DOFS_1D-1-a] = resOdd - resEven;
    }
  }
  
  __syncthreads();

  
  // test in 'b'
  if(t<NUM_DOFS_1D*NUM_QUAD_1D){
    const int a = t%NUM_DOFS_1D;
    const int k = t/NUM_DOFS_1D;
    
    for(int j=0;j<HALF_QUAD_1D;++j){
      dfloat ApOdd  = s_Ap[blk][k][j][a];
      dfloat ApEven = s_Ap[blk][k][NUM_QUAD_1D-1-j][a];
      r_tmpOdd[j]  = ApOdd + ApEven;
      r_tmpEven[j] = ApOdd - ApEven;
    }

    if(NUM_QUAD_1D%2)
      r_tmpOdd[HALF_QUAD_1D-1] *= 0.5f;    
    
#pragma unroll
    for(int b=0;b<HALF_DOFS_1D;++b){
      dfloat resOdd = 0, resEven = 0;
      
#pragma unroll
      for(int j=0;j<HALF_QUAD_1D;++j){
	int jb = ijN(b,j,HALF_DOFS_1D);
	resOdd  += oddDofToQuad[jb]*r_tmpOdd[j];
	resEven += evenDofToQuad[jb]*r_tmpEven[j];
      }
      
      s_Ap[blk][k][b][a]               = resOdd + resEven;
      s_Ap[blk][k][NUM_DOFS_1D-1-b][a] = resOdd - resEven;
    }
  }
  
  __syncthreads();

  // test in 'c'
  if(t<NUM_DOFS_2D){
    
    const int a = t%NUM_DOFS_1D;
    const int b = t/NUM_DOFS_1D;

    for(int k=0;k<HALF_QUAD_1D;++k){
      dfloat ApOdd  = s_Ap[blk][k][b][a];
      dfloat ApEven = s_Ap[blk][NUM_QUAD_1D-1-k][b][a];
      r_tmpOdd[k]  = ApOdd + ApEven;
      r_tmpEven[k] = ApOdd - ApEven;
    }

    if(NUM_QUAD_1D%2)
      r_tmpOdd[HALF_QUAD_1D-1] *= 0.5f;    
    
#pragma unroll
    for(int c=0;c<HALF_DOFS_1D;++c){
      dfloat resOdd = 0, resEven = 0;
      
#pragma unroll
      for(int k=0;k<HALF_QUAD_1D;++k){
	int kc = ijN(c,k,HALF_DOFS_1D);
	resOdd  += oddDofToQuad[kc]*r_tmpOdd[k];
	resEven += evenDofToQuad[kc]*r_tmpEven[k];
      }
      
      r_Ap[c]               = resOdd + resEven;
      r_Ap[NUM_DOFS_1D-1-c] = resOdd - resEven;
      
    }
  }

}

template <int NUM_DOFS_1D, int NUM_QUAD_1D, int p_Nblock >
 __global__ void BK3ConstantKernel(const int numElements,
				    const dfloat lambda,
				    const dfloat * __restrict__ op,
				    const dfloat * __restrict__ oddDofToQuad,
				    const dfloat * __restrict__ evenDofToQuad,
				    const dfloat * __restrict__ QuadToQuadD,
				    const dfloat * __restrict__ oddQuadToQuadD,
				    const dfloat * __restrict__ evenQuadToQuadD,
				    const dfloat * __restrict__ solIn,
				    dfloat * __restrict__ solOut){
  
  __shared__ dfloat s_tmp1[p_Nblock][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D+p_padCubNq];
  __shared__ dfloat s_QuadToQuadD[NUM_QUAD_2D];
  
  dfloat r_Aq[NUM_QUAD_1D];

  const unsigned int t = threadIdx.x;
  const int blk = threadIdx.y;
  
  const int element = blockIdx.x*p_Nblock + blk;
  
  const unsigned int a = t%NUM_DOFS_1D;
  const unsigned int b = t/NUM_DOFS_1D;

  s_QuadToQuadD[t] = QuadToQuadD[t];
  
  if(element < numElements){
    if(t<NUM_DOFS_2D){
      for(int c=0;c<NUM_DOFS_1D;++c){
	
	int id = ijklN(a,b,c,element,NUM_DOFS_1D);
	
	r_Aq[c] = solIn[id];
      }
    }
  }

  __syncthreads();
  
  BK3OuterDevice  <NUM_DOFS_1D, NUM_QUAD_1D, p_Nblock>
    (numElements, element, lambda, op, const_oddDofToQuad, const_evenDofToQuad, s_QuadToQuadD, const_oddQuadToQuadD, const_evenQuadToQuadD, s_tmp1, r_Aq);
  
  if(element<numElements){
    if(t<NUM_DOFS_2D){
#pragma unroll
      for(int c=0;c<NUM_DOFS_1D;++c){
	int id = ijklN(a,b,c,element,NUM_DOFS_1D);
	solOut[id] = r_Aq[c];
      }
    }
  }
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


void runMassMatrixMultiplyKernel(cudaStream_t stream, int Nq, int cubNq, int numElements, dfloat lambda,
				 dfloat *c_op,
				 dfloat *c_oddDofToQuad, dfloat *c_evenDofToQuad,
				 dfloat *c_QuadToQuadD, dfloat *c_oddQuadToQuadD, dfloat *c_evenQuadToQuadD,
				 dfloat *c_solIn, dfloat *c_solOut){
  
#define BK3Kernel(Nq,cubNq,Nblock)					\
  {									\
    dim3 G((numElements+Nblock-1)/Nblock, 1, 1);			\
    dim3 B(cubNq*cubNq, Nblock, 1);					\
    BK3ConstantKernel<Nq,cubNq,Nblock> <<< G, B, 0, stream >>>		\
      (numElements, lambda, c_op, c_oddDofToQuad, c_evenDofToQuad, c_QuadToQuadD, c_oddQuadToQuadD,c_evenQuadToQuadD, c_solIn, c_solOut); \
  }
  
#define ERR printf("BK3Register with Nq=%d, cubNq=%d not available", Nq, cubNq); exit(-1)

  int Nblock = 1;
  if(Nq==2){
    switch(cubNq){
    case 2: BK3Kernel(2,2,16); break;
    case 3: BK3Kernel(2,3, 7); break;
    case 4: BK3Kernel(2,4, 4); break;
    case 5: BK3Kernel(2,5, 5); break;
    case 6: BK3Kernel(2,6, 3); break;
    default: ERR;
    }
    return;
  }

  if(Nq==3){
    switch(cubNq){
    case 3: BK3Kernel(3,3,7); break;
    case 4: BK3Kernel(3,4,16); break;
    case 5: BK3Kernel(3,5,5); break;
    case 6: BK3Kernel(3,6,3); break;
    case 7: BK3Kernel(3,7,2); break;
    default: ERR;
    }
    return;
  }

  if(Nq==4){
    switch(cubNq){
    case 4: BK3Kernel(4,4,4); break;
    case 5: BK3Kernel(4,5,5); break;
    case 6: BK3Kernel(4,6,3); break;
    case 7: BK3Kernel(4,7,2); break;
    case 8: BK3Kernel(4,8,1); break;
    default: ERR;
    }
    return;
  }

  if(Nq==5){
    switch(cubNq){
    case 5: BK3Kernel(5,5,5); break;
    case 6: BK3Kernel(5,6,3); break;
    case 7: BK3Kernel(5,7,2); break;
    case 8: BK3Kernel(5,8,1); break;
    case 9: BK3Kernel(5,9,2); break;
    default: ERR;
    }
    return;
  }

  if(Nq==6){
    switch(cubNq){
    case 6:  BK3Kernel(6, 6, 3); break; // Nb=3 best so far
    case 7:  BK3Kernel(6, 7, 2); break;
    case 8:  BK3Kernel(6, 8, 1); break;
    case 9:  BK3Kernel(6, 9, 2); break;
    case 10: BK3Kernel(6,10, 1); break;
    default: ERR;
    }
    return;
  }

  if(Nq==7){
    switch(cubNq){
    case 7:  BK3Kernel(7, 7,2); break;
    case 8:  BK3Kernel(7, 8,1); break;
    case 9:  BK3Kernel(7, 9,2); break;
    case 10: BK3Kernel(7,10,1); break;
    case 11: BK3Kernel(7,11,1); break;

    default: ERR;
    }
    return;
  }

  if(Nq==8){
    switch(cubNq){
    case 8:  BK3Kernel(8, 8,1); break;
    case 9:  BK3Kernel(8, 9,2); break;
    case 10: BK3Kernel(8,10,1); break;
    case 11: BK3Kernel(8,11,1); break;
    case 12: BK3Kernel(8,12,1); break;
    default: ERR;
    }
    return;
  }

  if(Nq==9){
    switch(cubNq){
    case 9:  BK3Kernel(9, 9,1); break;
    case 10: BK3Kernel(9,10,1); break;
    case 11: BK3Kernel(9,11,1); break;
    case 12: BK3Kernel(9,12,1); break;
    case 13: BK3Kernel(9,13,1); break;

    default: ERR;
    }
    return;
  }

  if(Nq==10){
    switch(cubNq){
    case 10: BK3Kernel(10,10,1); break;
    case 11: BK3Kernel(10,11,1); break;
    case 12: BK3Kernel(10,12,1); break;
    case 13: BK3Kernel(10,13,1); break;
    case 14: BK3Kernel(10,14,1); break;
    default: ERR;
    }
    return;
  }

  if(Nq==11){
    switch(cubNq){
    case 11: BK3Kernel(11,11,1); break;
    case 12: BK3Kernel(11,12,1); break;
    case 13: BK3Kernel(11,13,1); break;
    case 14: BK3Kernel(11,14,1); break;
    case 15: BK3Kernel(11,15,1); break;

    default: ERR;
    }
    return;
  }
  
  if(Nq==12){
    switch(cubNq){
    case 12: BK3Kernel(12,12,1); break;
    case 13: BK3Kernel(12,13,1); break;
    case 14: BK3Kernel(12,14,1); break;
    case 15: BK3Kernel(12,15,1); break;
      //    case 16: BK3Kernel(12,16,1); break;
    default: ERR;
    }
    return;
  }

  if(Nq==13){
    switch(cubNq){
    case 13: BK3Kernel(13,13,1); break;
    case 14: BK3Kernel(13,14,1); break;
    case 15: BK3Kernel(14,15,1); break;
    case 16: BK3Kernel(15,16,1); break;
      //    case 16: BK3Kernel(12,16,1); break;
    default: ERR;
    }
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
    printf("Usage: ./BK3 Nq cubNq numElements\n");
    exit(-1);
  }

  // read number of elements
  int        Nq = atoi(argv[1]);
  int     cubNq = atoi(argv[2]);
  int numElements = atoi(argv[3]);

  dfloat lambda = 0;
  
  printf("Running: Nq=%d, cubNq=%d, numElements=%d\n", Nq, cubNq, numElements);

  if(cubNq<Nq){
    printf("cubNq must be > Nq\n");
    exit(-1);
  }
  
  int   Np = Nq*Nq*Nq;
  int   cubNp = cubNq*cubNq*cubNq;

  int halfNq = ((Nq+1)/2);
  int halfCubNq = ((cubNq+1)/2);

  int    Ntotal = numElements*Np;
  int cubNtotal = numElements*cubNp;

  int Ntests = 10;
  
  double estimatedActualDeviceBandwidth =
    bandwidthTest(stream, Ntests, (Ntotal*2+7*cubNtotal)*sizeof(dfloat));
  
  dfloat *h_op,      *c_op;
  dfloat *h_solOut,       *c_solOut;
  dfloat *h_solIn,        *c_solIn;
  dfloat *h_DofToQuad,    *c_DofToQuad;
  dfloat *c_oddDofToQuad, *c_evenDofToQuad;

  dfloat *h_QuadToQuadD,    *c_QuadToQuadD;
  dfloat *c_oddQuadToQuadD, *c_evenQuadToQuadD;

  // float fields
  randAlloc(cubNtotal*p_Nggeo, &h_op, &c_op);
  
  randAlloc(Ntotal, &h_solIn, &c_solIn);
  randAlloc(Ntotal, &h_solOut, &c_solOut);
  
  randAlloc(Nq*cubNq, &h_DofToQuad, &c_DofToQuad);
  randAlloc(cubNq*cubNq, &h_QuadToQuadD, &c_QuadToQuadD);

  // ------------------------------------------------------------------------------
  // build element nodes and operators
  
  dfloat *r, *w;
  dfloat *cubr, *cubw;
  
  meshJacobiGL(0,0,Nq-1, &r, &w);
  meshJacobiGQ(0,0,cubNq-1, &cubr, &cubw);
  
  meshDmatrix1D(cubNq-1, cubNq, cubr, &h_QuadToQuadD);

  meshInterpolationMatrix1D(Nq-1, Nq, r, cubNq, cubr, &h_DofToQuad);

  // ------------------------------------------------------------------------------
  // create Odd-even packed storage for I and transpose(I) and push to constant memory
  buildOddEvenMatrices (   Nq,cubNq, h_DofToQuad,   &c_DofToQuad,   &c_oddDofToQuad,   &c_evenDofToQuad  );
  buildOddEvenMatrices (cubNq,cubNq, h_QuadToQuadD, &c_QuadToQuadD, &c_oddQuadToQuadD, &c_evenQuadToQuadD);

  cudaMemcpyToSymbol(const_DofToQuad,     c_DofToQuad,    cubNq*Nq*sizeof(dfloat), 0, cudaMemcpyDeviceToDevice);
  cudaMemcpyToSymbol(const_oddDofToQuad,  c_oddDofToQuad, halfNq*halfCubNq*sizeof(dfloat), 0, cudaMemcpyDeviceToDevice);
  cudaMemcpyToSymbol(const_evenDofToQuad, c_evenDofToQuad, halfNq*halfCubNq*sizeof(dfloat), 0, cudaMemcpyDeviceToDevice);
  
  cudaMemcpyToSymbol(const_QuadToQuadD,     c_QuadToQuadD,     cubNq*cubNq*sizeof(dfloat), 0, cudaMemcpyDeviceToDevice);
  cudaMemcpyToSymbol(const_oddQuadToQuadD,  c_oddQuadToQuadD,  halfCubNq*halfCubNq*sizeof(dfloat), 0, cudaMemcpyDeviceToDevice);
  cudaMemcpyToSymbol(const_evenQuadToQuadD, c_evenQuadToQuadD, halfCubNq*halfCubNq*sizeof(dfloat), 0, cudaMemcpyDeviceToDevice);
  
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);	

  // KERNEL GRID
  // do nothing kernel test
  dfloat nothingElapsed = nothingTest(stream, Ntests);
  
  // warm up call
  runMassMatrixMultiplyKernel (stream, Nq, cubNq, numElements, lambda,
			       c_op,
			       c_oddDofToQuad, c_evenDofToQuad,
			       c_QuadToQuadD, c_oddQuadToQuadD, c_evenQuadToQuadD,
			       c_solIn, c_solOut);

#if USE_GRAPH==1
  // cuda stream capture
  cudaGraph_t graph;
  
  cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

  for(int test=0;test<Ntests;++test){

    runMassMatrixMultiplyKernel (stream, Nq, cubNq, numElements, lambda,
				 c_op,
				 c_oddDofToQuad, c_evenDofToQuad,
				 c_QuadToQuadD, c_oddQuadToQuadD, c_evenQuadToQuadD,
				 c_solIn, c_solOut);
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

      runMassMatrixMultiplyKernel (stream, Nq, cubNq, numElements, lambda,
				   c_op,
				   c_oddDofToQuad, c_evenDofToQuad,
				   c_QuadToQuadD, c_oddQuadToQuadD, c_evenQuadToQuadD,
				   c_solIn, c_solOut);
      
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

    int bytesMoved = (2*Np+7*cubNp)*sizeof(dfloat); // x, Mx, opa   
    double bw = (bytesMoved*numElements/elapsed)/1.e9;
    
    printf("%2d %8d %8d %e %e %e %e %e %%%% [MassMatrixMultiply: N, numElements, Ndofs,"
	   " elapsed, dofsPerSecond, nothingElapsed, BW in GB/s, estimatedActualDeviceBandwidth]\n",
	   Nq-1, numElements, Np*numElements, elapsed, numElements*(Np/elapsed),
	   nothingElapsed, bw, estimatedActualDeviceBandwidth);
  }

  // check output is correct
  meshReferenceBK3(Nq, cubNq, numElements, lambda, h_op, h_DofToQuad, h_QuadToQuadD, h_solIn, h_solOut);

  // copy device version to host old q
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
