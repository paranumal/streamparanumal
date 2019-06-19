/*

See LICENSE file.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "meshBasis.hpp"

#define READ_TO_REGISTER 1

#define CUDA_SYMBOL(a) a


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

#define MAX_DOFS_1D 16
#define MAX_QUAD_1D 16

#define MAX_HALF_DOFS_1D 8
#define MAX_HALF_QUAD_1D 8

#define HALF_DOFS_1D ((NUM_DOFS_1D+1)/2)
#define HALF_QUAD_1D ((NUM_QUAD_1D+1)/2)

#define p_padCubNq 0
// ((NUM_QUAD_1D%4) ? 0:1)

#define NUM_DOFS_2D (NUM_DOFS_1D*NUM_DOFS_1D)
#define NUM_DOFS_3D (NUM_DOFS_1D*NUM_DOFS_1D*NUM_DOFS_1D)

#define NUM_QUAD_2D (NUM_QUAD_1D*NUM_QUAD_1D)
#define NUM_QUAD_3D (NUM_QUAD_1D*NUM_QUAD_1D*NUM_QUAD_1D)

#define p_Nvgeo 1
#define p_JWID 0

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

__constant__ dfloat const_DofToQuad[MAX_QUAD_1D*MAX_DOFS_1D];
__constant__ dfloat const_oddDofToQuad[MAX_HALF_QUAD_1D*MAX_HALF_DOFS_1D];
__constant__ dfloat const_evenDofToQuad[MAX_HALF_QUAD_1D*MAX_HALF_DOFS_1D];

void randAlloc(int N, dfloat **h_a, dfloat **c_a){

  *h_a = (dfloat*) calloc(N, sizeof(dfloat));

  for(int n=0;n<N;++n)
    h_a[0][n] = drand48();

  cudaMalloc(c_a, N*sizeof(dfloat));

  cudaMemcpy(c_a[0], h_a[0], N*sizeof(dfloat), cudaMemcpyHostToDevice);

}

__global__ void nothingKernel(){  }

__global__ void nothingVerboseKernel(int n, dfloat *creOut, dfloat *cimOut){


  if(n==-1 || n==-7098 || n==1023 || n==3521){ // this will never be true

    dfloat cre = threadIdx.x + blockIdx.x*blockDim.x;
    dfloat cim = threadIdx.y + blockIdx.x*blockDim.y;

#pragma unroll 1
    for(int i=0;i<1;++i){
      dfloat tmpre = cre*cre-cim*cim;
      dfloat tmpim = 2.*cre*cim;

      cre = tmpre;
      cim = tmpim;

    }

    creOut[0] = cre;
    
  }
}

template <int NUM_DOFS_1D, int NUM_QUAD_1D, int p_Nblock >
  __forceinline__ __device__ 
  void BK1MonolithicDevice(const int numElements,
			   const int element,
			   const dfloat * __restrict__ op,
			   const dfloat * __restrict__ DofToQuad,
			   dfloat s_Ap[p_Nblock][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D+p_padCubNq],
			   dfloat * __restrict__ r_Ap){
  
  dfloat r_tmp[NUM_QUAD_1D];
  
  const int t   = threadIdx.x;
  const int blk = threadIdx.y;
  
  // assumes barrier before s_Ap was used last
  // TRY REARRANING THIS
  
  // transform in 'c'
  if(t<NUM_DOFS_2D){
    const int a = t%NUM_DOFS_1D;
    const int b = t/NUM_DOFS_1D;
    
#pragma unroll
    for(int k=0;k<NUM_QUAD_1D;++k){
      dfloat res = 0;
      
#pragma unroll
      for(int c=0;c<NUM_DOFS_1D;++c){
	int kc = ijN(c,k,NUM_DOFS_1D);		
	res  += DofToQuad[kc]*r_Ap[c];
      }
      
      s_Ap[blk][k][b][a]  = res;
    }
  }
  
  __syncthreads();

  // transform in 'b'
  if(t<NUM_DOFS_1D*NUM_QUAD_1D){
    const int a = t%NUM_DOFS_1D;
    const int k = t/NUM_DOFS_1D;
    
#pragma unroll
    for(int b=0;b<NUM_DOFS_1D;++b){
      r_tmp[b]  = s_Ap[blk][k][b][a];
    }
    
#pragma unroll
    for(int j=0;j<NUM_QUAD_1D;++j){
      dfloat res = 0;
      
#pragma unroll
      for(int b=0;b<NUM_DOFS_1D;++b){
	int jb = ijN(b,j,NUM_DOFS_1D);
	res  += DofToQuad[jb]*r_tmp[b];
      }
      s_Ap[blk][k][j][a] = res;
    }
  }
  
  __syncthreads();

  // transform in 'a'
  {
    const int j = t%NUM_QUAD_1D;
    const int k = t/NUM_QUAD_1D;
    
#pragma unroll
    for(int a=0;a<NUM_DOFS_1D;++a){
      r_tmp[a]  = s_Ap[blk][k][j][a];
    }
    
#pragma unroll
    for(int i=0;i<NUM_QUAD_1D;++i){
      dfloat res = 0;
      
#pragma unroll
      for(int a=0;a<NUM_DOFS_1D;++a){
	int ia = ijN(a,i,NUM_DOFS_1D);
	res  += DofToQuad[ia]*r_tmp[a];
      }
	
      int gid = ijklN(i,j,k,element, NUM_QUAD_1D);
      
      dfloat WJ = (element<numElements) ? op[gid]: 0;
      
      r_Ap[i] = WJ*res;
    }
    
#pragma unroll
    for(int a=0;a<NUM_DOFS_1D;++a){
      dfloat res = 0;
      
#pragma unroll
      for(int i=0;i<NUM_QUAD_1D;++i){
	int ia = ijN(a,i,NUM_DOFS_1D);
	res  += DofToQuad[ia]*r_Ap[i];
      }
    
      s_Ap[blk][k][j][a] = res;
    }
  }
  
  __syncthreads();

  
  // test in 'b'
  if(t<NUM_DOFS_1D*NUM_QUAD_1D){
    const int a = t%NUM_DOFS_1D;
    const int k = t/NUM_DOFS_1D;
    
    for(int j=0;j<NUM_QUAD_1D;++j){
      r_tmp[j]  = s_Ap[blk][k][j][a];
    }
    
#pragma unroll
    for(int b=0;b<NUM_DOFS_1D;++b){
      dfloat res = 0;
      
#pragma unroll
      for(int j=0;j<NUM_QUAD_1D;++j){
	int jb = ijN(b,j,NUM_DOFS_1D);
	res += DofToQuad[jb]*r_tmp[j];
      }
      
      s_Ap[blk][k][b][a] = res;
    }
  }
  
  __syncthreads();

  // test in 'c'
  if(t<NUM_DOFS_2D){
    const int a = t%NUM_DOFS_1D;
    const int b = t/NUM_DOFS_1D;
    
    for(int k=0;k<NUM_QUAD_1D;++k){
      r_tmp[k]  = s_Ap[blk][k][b][a];
    }

#pragma unroll
    for(int c=0;c<NUM_DOFS_1D;++c){
      dfloat res = 0; 
      
#pragma unroll
      for(int k=0;k<NUM_QUAD_1D;++k){
	int kc = ijN(c,k,NUM_DOFS_1D);
	res += DofToQuad[kc]*r_tmp[k];
      }

      r_Ap[c] = res;
    }
  }

#if USE_CONTIGUOUS_OUTPUT==1
  
  __syncthreads();

  // write to shared
  if(t<NUM_DOFS_2D){

#pragma unroll
    for(int c=0;c<NUM_DOFS_1D;++c){
      const int a = t%NUM_DOFS_1D;
      const int b = t/NUM_DOFS_1D;
      int id = ijklN(a,b,c,blk, NUM_DOFS_1D);
      s_Ap[0][0][0][id] = r_Ap[c];
    }
  }

#endif
  __syncthreads();
  
}

template <int NUM_DOFS_1D, int NUM_QUAD_1D, int p_Nblock >
  __forceinline__ __device__ 
  void BK1OddEvenDevice(const int numElements,
			const int element,
			const dfloat * __restrict__ op,
			const dfloat * __restrict__ oddDofToQuad,
			const dfloat * __restrict__ evenDofToQuad,
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
      
      s_Ap[blk][NUM_QUAD_1D-1-k][b][a] = resOdd - resEven;
      s_Ap[blk][k][b][a]               = resOdd + resEven;
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
      
      s_Ap[blk][k][NUM_QUAD_1D-1-j][a] = resOdd-resEven;
      s_Ap[blk][k][j][a]               = resOdd+resEven;
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
      
      int gid1 = ijklN(NUM_QUAD_1D-1-i,j,k,element, NUM_QUAD_1D);
      int gid2 = ijklN(i,j,k,element, NUM_QUAD_1D);

      dfloat WJ1 = (element<numElements) ? op[gid1]:0;
      dfloat WJ2 = (element<numElements) ? op[gid2]:0;

#if 0
      s_Ap[blk][k][j][NUM_QUAD_1D-1-i] = WJ1*(resOdd-resEven);
      s_Ap[blk][k][j][i]               = WJ2*(resOdd+resEven);
#else
      r_Ap[NUM_QUAD_1D-1-i] = WJ1*(resOdd-resEven);
      r_Ap[i]               = WJ2*(resOdd+resEven);
#endif
    }
  }
  
  __syncthreads();
  
  {
    const int j = t%NUM_QUAD_1D;
    const int k = t/NUM_QUAD_1D;
    
#pragma unroll
    for(int i=0;i<HALF_QUAD_1D;++i){
#if 0
      dfloat ApOdd  = s_Ap[blk][k][j][i];
      dfloat ApEven = s_Ap[blk][k][j][NUM_QUAD_1D-1-i];
#else
      dfloat ApOdd  = r_Ap[i];
      dfloat ApEven = r_Ap[NUM_QUAD_1D-1-i];
#endif
      
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
      
      s_Ap[blk][k][j][NUM_DOFS_1D-1-a] = resOdd-resEven;
      s_Ap[blk][k][j][a]               = resOdd+resEven;
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
      
      s_Ap[blk][k][NUM_DOFS_1D-1-b][a] = resOdd - resEven;
      s_Ap[blk][k][b][a]               = resOdd + resEven;
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
      
      r_Ap[NUM_DOFS_1D-1-c] = resOdd - resEven;
      r_Ap[c]               = resOdd + resEven;
    }
  }

#if USE_CONTIGUOUS_OUTPUT==1
  __syncthreads();

  // write to shared
  if(t<NUM_DOFS_2D){

    const int a = t%NUM_DOFS_1D;
    const int b = t/NUM_DOFS_1D;
    
#pragma unroll
    for(int c=0;c<NUM_DOFS_1D;++c){
      int id = ijklN(a,b,c,blk, NUM_DOFS_1D);
      s_Ap[0][0][0][id] = r_Ap[c];
    }
  }
#endif
  
  __syncthreads();

}


template <int NUM_DOFS_1D, int NUM_QUAD_1D, int p_Nblock >
  __global__ void BK1RegisterKernel(const int numElements,
				    const dfloat * __restrict__ op,
				    const dfloat * __restrict__ oddDofToQuad,
				    const dfloat * __restrict__ evenDofToQuad,
				    const dfloat * __restrict__ solIn,
				    dfloat * __restrict__ solOut){
  
  __shared__ dfloat s_tmp1[p_Nblock][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D+p_padCubNq];

  dfloat r_oddDofToQuad[HALF_QUAD_1D*HALF_DOFS_1D];
  dfloat r_evenDofToQuad[HALF_QUAD_1D*HALF_DOFS_1D];

  dfloat r_Aq[NUM_QUAD_1D];

  const unsigned int t = threadIdx.x;
  const int blk = threadIdx.y;
  
  const int element = blockIdx.x*p_Nblock + blk;
  
  const unsigned int a = t%NUM_DOFS_1D;
  const unsigned int b = t/NUM_DOFS_1D;

  
#if READ_TO_REGISTER==1
  if(element < numElements && t<NUM_DOFS_2D){
    for(int c=0;c<NUM_DOFS_1D;++c){
      int id = ijklN(a,b,c,element, NUM_DOFS_1D); 
      
      r_Aq[c] = solIn[id];
    }
  }
#else
  int n = t + blk*NUM_QUAD_2D;
  
  while(n<p_Nblock*NUM_DOFS_3D){
    
    int id = n + blockIdx.x*p_Nblock*NUM_DOFS_3D;
    if(id<numElements*NUM_DOFS_3D){
      s_tmp1[0][0][0][n] = solIn[id];
    }
    n+=NUM_QUAD_2D*p_Nblock;
  }
  
#endif
  {
    __shared__ dfloat s_oddDofToQuad[HALF_DOFS_1D*HALF_QUAD_1D];
    __shared__ dfloat s_evenDofToQuad[HALF_QUAD_1D*HALF_DOFS_1D];
    
    if(blk==0)
      for(int n=t;n<HALF_DOFS_1D*HALF_QUAD_1D;n+=NUM_QUAD_2D){
	s_oddDofToQuad[n] = oddDofToQuad[n];
	s_evenDofToQuad[n] = evenDofToQuad[n];
      }
    
    __syncthreads();
    
    // now copy shared data to thread local register arrays
    for(int n=0;n<HALF_DOFS_1D*HALF_QUAD_1D;++n){
      r_oddDofToQuad[n] = s_oddDofToQuad[n];
      r_evenDofToQuad[n] = s_evenDofToQuad[n];
    }
  }

#if READ_TO_REGISTER==0
  if(t<NUM_DOFS_2D)
    for(int c=0;c<NUM_DOFS_1D;++c)
      r_Aq[c] = s_tmp1[blk][c][b][a];

  __syncthreads();
#endif
  
  BK1OddEvenDevice <NUM_DOFS_1D, NUM_QUAD_1D, p_Nblock>
    (numElements, element, op, r_oddDofToQuad, r_evenDofToQuad, s_tmp1, r_Aq);

#if USE_CONTIGUOUS_OUTPUT==0
  if(element<numElements && t<NUM_DOFS_2D){
#pragma unroll
    for(int c=0;c<NUM_DOFS_1D;++c){
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      solOut[id] = r_Aq[c];
    }
  }
#else
  
  int n = t + blk*NUM_QUAD_2D;
  
  while(n<p_Nblock*NUM_DOFS_3D){
    
    int id = n + blockIdx.x*p_Nblock*NUM_DOFS_3D;
    if(id<numElements*NUM_DOFS_3D){
      solOut[id] = s_tmp1[0][0][0][n];
    }
    n+=NUM_QUAD_2D*p_Nblock;
  }
#endif  
}


template <int NUM_DOFS_1D, int NUM_QUAD_1D, int p_Nblock >
  __global__ void BK1SharedKernel(const int numElements,
				  const dfloat * __restrict__ op,
				  const dfloat * __restrict__ oddDofToQuad,
				  const dfloat * __restrict__ evenDofToQuad,
				  const dfloat * __restrict__ solIn,
				  dfloat * __restrict__ solOut){
  
  __shared__ dfloat s_tmp1[p_Nblock][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D+p_padCubNq];
  __shared__ dfloat s_oddDofToQuad[HALF_QUAD_1D*HALF_DOFS_1D];
  __shared__ dfloat s_evenDofToQuad[HALF_QUAD_1D*HALF_DOFS_1D];

  dfloat r_Aq[NUM_QUAD_1D];

  const unsigned int t = threadIdx.x;
  const int blk = threadIdx.y;
  
  const int element = blockIdx.x*p_Nblock + blk;
  
  const unsigned int a = t%NUM_DOFS_1D;
  const unsigned int b = t/NUM_DOFS_1D;

#if READ_TO_REGISTER==1
  if(element < numElements && t<NUM_DOFS_2D){
    for(int c=0;c<NUM_DOFS_1D;++c){
      
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      
      r_Aq[c] = solIn[id];
    }
  }
#else
  int n = t + blk*NUM_QUAD_2D;
  
  while(n<p_Nblock*NUM_DOFS_3D){
    
    int id = n + blockIdx.x*p_Nblock*NUM_DOFS_3D;
    if(id<numElements*NUM_DOFS_3D){
      s_tmp1[0][0][0][n] = solIn[id];
    }
    n+=NUM_QUAD_2D*p_Nblock;
  }
#endif
  
  if(blk==0)
    for(int n=t;n<HALF_DOFS_1D*HALF_QUAD_1D;n+=NUM_QUAD_2D){
      s_oddDofToQuad[n] = oddDofToQuad[n];
      s_evenDofToQuad[n] = evenDofToQuad[n];
    }

  __syncthreads();

#if READ_TO_REGISTER==0
  if(t<NUM_DOFS_2D)
    for(int c=0;c<NUM_DOFS_1D;++c)
      r_Aq[c] = s_tmp1[blk][c][b][a];

  __syncthreads();
#endif
  
  BK1OddEvenDevice  <NUM_DOFS_1D, NUM_QUAD_1D, p_Nblock>
    (numElements, element, op, s_oddDofToQuad, s_evenDofToQuad, s_tmp1, r_Aq);

#if USE_CONTIGUOUS_OUTPUT==0
  if(element<numElements && t<NUM_DOFS_2D){
#pragma unroll
    for(int c=0;c<NUM_DOFS_1D;++c){
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      solOut[id] = r_Aq[c];
    }
  }
#else
  
  int n = t + blk*NUM_QUAD_2D;
  
  while(n<p_Nblock*NUM_DOFS_3D){
    
    int id = n + blockIdx.x*p_Nblock*NUM_DOFS_3D;
    if(id<numElements*NUM_DOFS_3D){
      solOut[id] = s_tmp1[0][0][0][n];
    }
    n+=NUM_QUAD_2D*p_Nblock;
  }
#endif  
}

template <int NUM_DOFS_1D, int NUM_QUAD_1D, int p_Nblock >
  __global__ void BK1ConstantKernel(const int numElements,
				    const dfloat * __restrict__ op,
				    const dfloat * __restrict__ oddDofToQuad,
				    const dfloat * __restrict__ evenDofToQuad,
				    const dfloat * __restrict__ solIn,
				    dfloat * __restrict__ solOut){
  
  __shared__ dfloat s_tmp1[p_Nblock][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D+p_padCubNq];

  dfloat r_Aq[NUM_QUAD_1D];

  const unsigned int t = threadIdx.x;
  const int blk = threadIdx.y;
  
  const int element = blockIdx.x*p_Nblock + blk;
  
  const unsigned int a = t%NUM_DOFS_1D;
  const unsigned int b = t/NUM_DOFS_1D;

#if READ_TO_REGISTER==1
  if(element < numElements && t<NUM_DOFS_2D){
    for(int c=0;c<NUM_DOFS_1D;++c){
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      
      r_Aq[c] = solIn[id];
    }
  }
#else
  int n = t + blk*NUM_QUAD_2D;
  
  while(n<p_Nblock*NUM_DOFS_3D){
    
    int id = n + blockIdx.x*p_Nblock*NUM_DOFS_3D;
    if(id<numElements*NUM_DOFS_3D){
      s_tmp1[0][0][0][n] = solIn[id];
    }
    n+=NUM_QUAD_2D*p_Nblock;
  }
#endif

  __syncthreads();

#if READ_TO_REGISTER==0
  if(t<NUM_DOFS_2D)
    for(int c=0;c<NUM_DOFS_1D;++c)
      r_Aq[c] = s_tmp1[blk][c][b][a];

  __syncthreads();
#endif
  
  BK1OddEvenDevice  <NUM_DOFS_1D, NUM_QUAD_1D, p_Nblock>
    (numElements, element, op, const_oddDofToQuad, const_evenDofToQuad, s_tmp1, r_Aq);

#if USE_CONTIGUOUS_OUTPUT==0
  
  if(element<numElements && t<NUM_DOFS_2D){
#pragma unroll
    for(int c=0;c<NUM_DOFS_1D;++c){
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      solOut[id] = r_Aq[c];
    }
  }

#else
  
  int n = t + blk*NUM_QUAD_2D;
  
  while(n<p_Nblock*NUM_DOFS_3D){
    
    int id = n + blockIdx.x*p_Nblock*NUM_DOFS_3D;
    if(id<numElements*NUM_DOFS_3D){
      solOut[id] = s_tmp1[0][0][0][n];
    }
    n+=NUM_QUAD_2D*p_Nblock;
  }
#endif  

  
}

template <int NUM_DOFS_1D, int NUM_QUAD_1D, int p_Nblock >
  __global__ void BK1GlobalKernel(const int numElements,
				  const dfloat * __restrict__ op,
				  const dfloat * __restrict__ oddDofToQuad,
				  const dfloat * __restrict__ evenDofToQuad,
				  const dfloat * __restrict__ solIn,
				  dfloat * __restrict__ solOut){
  
  __shared__ dfloat s_tmp1[p_Nblock][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D+p_padCubNq];

  dfloat r_Aq[NUM_QUAD_1D];

  const unsigned int t = threadIdx.x;
  const int blk = threadIdx.y;
  
  const int element = blockIdx.x*p_Nblock + blk;
  
  const unsigned int a = t%NUM_DOFS_1D;
  const unsigned int b = t/NUM_DOFS_1D;

#if READ_TO_REGISTER==1
  if(element < numElements && t<NUM_DOFS_2D){
    for(int c=0;c<NUM_DOFS_1D;++c){
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      
      r_Aq[c] = solIn[id];
    }
  }
#else
  int n = t + blk*NUM_QUAD_2D;
  
  while(n<p_Nblock*NUM_DOFS_3D){
    
    int id = n + blockIdx.x*p_Nblock*NUM_DOFS_3D;
    if(id<numElements*NUM_DOFS_3D){
      s_tmp1[0][0][0][n] = solIn[id];
    }
    n+=NUM_QUAD_2D*p_Nblock;
  }
#endif

  __syncthreads();

#if READ_TO_REGISTER==0
  if(t<NUM_DOFS_2D)
    for(int c=0;c<NUM_DOFS_1D;++c)
      r_Aq[c] = s_tmp1[blk][c][b][a];

  __syncthreads();
#endif
  
  BK1OddEvenDevice  <NUM_DOFS_1D, NUM_QUAD_1D, p_Nblock>
    (numElements, element, op, oddDofToQuad, evenDofToQuad, s_tmp1, r_Aq);

#if USE_CONTIGUOUS_OUTPUT==0
  
  if(element<numElements && t<NUM_DOFS_2D){
#pragma unroll
    for(int c=0;c<NUM_DOFS_1D;++c){
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      solOut[id] = r_Aq[c];
    }
  }

#else
  
  int n = t + blk*NUM_QUAD_2D;
  
  while(n<p_Nblock*NUM_DOFS_3D){
    
    int id = n + blockIdx.x*p_Nblock*NUM_DOFS_3D;
    if(id<numElements*NUM_DOFS_3D){
      solOut[id] = s_tmp1[0][0][0][n];
    }
    n+=NUM_QUAD_2D*p_Nblock;
  }
#endif  

  
}



template <int NUM_DOFS_1D, int NUM_QUAD_1D, int p_Nblock >
  __global__ void BK1MonolithicGlobalKernel(const int numElements,
					    const dfloat * __restrict__ op,
					    const dfloat * __restrict__ DofToQuad,
					    const dfloat * __restrict__ evenDofToQuad,
					    const dfloat * __restrict__ solIn,
					    dfloat * __restrict__ solOut){
  
  __shared__ dfloat s_tmp1[p_Nblock][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D+p_padCubNq];

  dfloat r_Aq[NUM_QUAD_1D];

  const unsigned int t = threadIdx.x;
  const int blk = threadIdx.y;
  
  const int element = blockIdx.x*p_Nblock + blk;
  
  const unsigned int a = t%NUM_DOFS_1D;
  const unsigned int b = t/NUM_DOFS_1D;

#if READ_TO_REGISTER==1
  if(element < numElements){
    for(int c=0;c<NUM_DOFS_1D;++c){
      
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      
      r_Aq[c] = solIn[id];
    }
  }
#else
  int n = t + blk*NUM_QUAD_2D;
  
  while(n<p_Nblock*NUM_DOFS_3D){
    
    int id = n + blockIdx.x*p_Nblock*NUM_DOFS_3D;
    if(id<numElements*NUM_DOFS_3D){
      s_tmp1[0][0][0][n] = solIn[id];
    }
    n+=NUM_QUAD_2D*p_Nblock;
  }
#endif
  
  __syncthreads();

#if READ_TO_REGISTER==0
  if(t<NUM_DOFS_2D)
    for(int c=0;c<NUM_DOFS_1D;++c)
      r_Aq[c] = s_tmp1[blk][c][b][a];

  __syncthreads();
#endif
  
  BK1MonolithicDevice  <NUM_DOFS_1D, NUM_QUAD_1D, p_Nblock>
    (numElements, element, op, DofToQuad, s_tmp1, r_Aq);

#if USE_CONTIGUOUS_OUTPUT==0

  if(element<numElements && t<NUM_DOFS_2D){
#pragma unroll
    for(int c=0;c<NUM_DOFS_1D;++c){
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      solOut[id] = r_Aq[c];
    }
  }

#else
  
  int n = t + blk*NUM_QUAD_2D;
  
  while(n<p_Nblock*NUM_DOFS_3D){
    
    int id = n + blockIdx.x*p_Nblock*NUM_DOFS_3D;
    if(id<numElements*NUM_DOFS_3D){
      solOut[id] = s_tmp1[0][0][0][n];
    }
    n+=NUM_QUAD_2D*p_Nblock;
  }
#endif  

  
}


template <int NUM_DOFS_1D, int NUM_QUAD_1D, int p_Nblock >
  __global__ void BK1MonolithicConstantKernel(const int numElements,
					      const dfloat * __restrict__ op,
					      const dfloat * __restrict__ DofToQuad,
					      const dfloat * __restrict__ evenDofToQuad,
					      const dfloat * __restrict__ solIn,
					      dfloat * __restrict__ solOut){
  
  __shared__ dfloat s_tmp1[p_Nblock][NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D+p_padCubNq];
  
  dfloat r_Aq[NUM_QUAD_1D];

  const unsigned int t = threadIdx.x;
  const int blk = threadIdx.y;
  
  const int element = blockIdx.x*p_Nblock + blk;
  
  const unsigned int a = t%NUM_DOFS_1D;
  const unsigned int b = t/NUM_DOFS_1D;

#if READ_TO_REGISTER==1
  if(element < numElements){
    for(int c=0;c<NUM_DOFS_1D;++c){
      
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      
      r_Aq[c] = solIn[id];
    }
  }
#else
  int n = t + blk*NUM_QUAD_2D;
  
  while(n<p_Nblock*NUM_DOFS_3D){
    
    int id = n + blockIdx.x*p_Nblock*NUM_DOFS_3D;
    if(id<numElements*NUM_DOFS_3D){
      s_tmp1[0][0][0][n] = solIn[id];
    }
    n+=NUM_QUAD_2D*p_Nblock;
  }
#endif
  
  __syncthreads();

#if READ_TO_REGISTER==0
  if(t<NUM_DOFS_2D)
    for(int c=0;c<NUM_DOFS_1D;++c)
      r_Aq[c] = s_tmp1[blk][c][b][a];

  __syncthreads();
#endif
  
  BK1MonolithicDevice  <NUM_DOFS_1D, NUM_QUAD_1D, p_Nblock>
    (numElements, element, op, const_DofToQuad, s_tmp1, r_Aq);

#if USE_CONTIGUOUS_OUTPUT==0

  if(element<numElements && t<NUM_DOFS_2D){
#pragma unroll
    for(int c=0;c<NUM_DOFS_1D;++c){
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      solOut[id] = r_Aq[c];
    }
  }

#else
  
  int n = t + blk*NUM_QUAD_2D;
  
  while(n<p_Nblock*NUM_DOFS_3D){
    
    int id = n + blockIdx.x*p_Nblock*NUM_DOFS_3D;
    if(id<numElements*NUM_DOFS_3D){
      solOut[id] = s_tmp1[0][0][0][n];
    }
    n+=NUM_QUAD_2D*p_Nblock;
  }
#endif  

  
}

template <int NUM_DOFS_1D, int NUM_QUAD_1D>
  __forceinline__ __device__ 
  void BK1CubeDevice(const int numElements,
		     const int element,
		     const dfloat * __restrict__ op,
		     const dfloat * __restrict__ DofToQuad,
		     dfloat s_p[NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D+p_padCubNq]){
  
  const int a = threadIdx.x;
  const int b = threadIdx.y;
  const int c = threadIdx.z;

  // assume s_p preloaded
  
  __syncthreads();
  
  // transform in 'a'
  dfloat res = 0;
  if(b<NUM_DOFS_1D && c<NUM_DOFS_1D){
#pragma unroll 
    for(int n=0;n<NUM_DOFS_1D;++n){
      int an = ijN(n,a,NUM_DOFS_1D);		
      res  += DofToQuad[an]*s_p[c][b][n];
    }
  }
  
  __syncthreads();
  
  if(b<NUM_DOFS_1D && c<NUM_DOFS_1D){
    s_p[c][b][a]  = res;
  }
  
  __syncthreads();
  
  // transform in 'b'
  res = 0;
  
  if(c<NUM_DOFS_1D){
#pragma unroll 
    for(int n=0;n<NUM_DOFS_1D;++n){
      int bn = ijN(n,b,NUM_DOFS_1D);		
      res  += DofToQuad[bn]*s_p[c][n][a];
    }
  }
  
  __syncthreads();

  if(c<NUM_DOFS_1D){
    s_p[c][b][a]  = res;
  }
  
  __syncthreads();

  int gid = ijklN(a,b,c,element, NUM_QUAD_1D);
  dfloat WJ = op[gid];
  
  // transform in 'c'
  res = 0;

#pragma unroll 
  for(int n=0;n<NUM_DOFS_1D;++n){
    int cn = ijN(n,c,NUM_DOFS_1D);		
    res  += DofToQuad[cn]*s_p[n][b][a];
  }

  __syncthreads();
  
  s_p[c][b][a] = WJ*res;
  
  __syncthreads();
  
  // test in 'c'
  res = 0;
  if(c<NUM_DOFS_1D){
#pragma unroll 
    for(int n=0;n<NUM_QUAD_1D;++n){
      int cn = ijN(c,n,NUM_DOFS_1D);		
      res  += DofToQuad[cn]*s_p[n][b][a];
    }
  }

  __syncthreads();

  if(c<NUM_DOFS_1D){
    s_p[c][b][a] = res;
  }

  __syncthreads();


  // transform in 'b'
  res = 0;
  if(b<NUM_DOFS_1D && c<NUM_DOFS_1D){
#pragma unroll 
    for(int n=0;n<NUM_QUAD_1D;++n){
      int bn = ijN(b,n,NUM_DOFS_1D);		
      res  += DofToQuad[bn]*s_p[c][n][a];
    }
  }
  
  __syncthreads();
  
  if(b<NUM_DOFS_1D && c<NUM_DOFS_1D){
    s_p[c][b][a]  = res;
  }

  __syncthreads();
  
  // test in 'a'
  res = 0;
  if(a<NUM_DOFS_1D && b<NUM_DOFS_1D && c<NUM_DOFS_1D){
#pragma unroll 
    for(int n=0;n<NUM_QUAD_1D;++n){
      int an = ijN(a,n,NUM_DOFS_1D);		
      res  += DofToQuad[an]*s_p[c][b][n];
    }
  }

  __syncthreads();
  
  if(a<NUM_DOFS_1D && b<NUM_DOFS_1D && c<NUM_DOFS_1D){
    s_p[c][b][a] = res;
  }

  __syncthreads();
}


template <int NUM_DOFS_1D, int NUM_QUAD_1D>
  __global__ void BK1CubeKernel(const int numElements,
				const dfloat * __restrict__ op,
				const dfloat * __restrict__ DofToQuad,
				const dfloat * __restrict__ solIn,
				dfloat * __restrict__ solOut){
  
  __shared__ dfloat s_p[NUM_QUAD_1D][NUM_QUAD_1D][NUM_QUAD_1D+p_padCubNq];
  __shared__ dfloat s_DofToQuad[NUM_QUAD_1D*NUM_DOFS_1D];
  
  const int element = blockIdx.x;
  
  const int i = threadIdx.x;
  const int j = threadIdx.y;
  const int k = threadIdx.z;

  int t = i + j*NUM_QUAD_1D + k*NUM_QUAD_2D;

  int a = t%NUM_DOFS_1D;
  int b = (t/NUM_DOFS_1D)%NUM_DOFS_1D;
  int c = (t/NUM_DOFS_2D);

  int id = ijklN(a,b,c,element,NUM_DOFS_1D);
    
  if(t<NUM_DOFS_3D){
    s_p[c][b][a] = solIn[t + element*NUM_DOFS_3D];
  }

  if(t<NUM_DOFS_1D*NUM_QUAD_1D){
    s_DofToQuad[t] = DofToQuad[t];
  }
  
  BK1CubeDevice  <NUM_DOFS_1D, NUM_QUAD_1D>
    (numElements, element, op, s_DofToQuad, s_p);
  
  if(t<NUM_DOFS_3D){
    solOut[t + element*NUM_DOFS_3D] = s_p[c][b][a] ;
  }
}




void buildInterpMatrices(int NUM_DOFS_1D, int NUM_QUAD_1D,
			 dfloat *h_DofToQuad,     dfloat *h_oddDofToQuad, dfloat *h_evenDofToQuad,
			 dfloat **c_oddDofToQuad, dfloat **c_evenDofToQuad){

  dfloat *X = (dfloat*) calloc(NUM_DOFS_1D*NUM_DOFS_1D, sizeof(dfloat));
  dfloat *invX = (dfloat*) calloc(NUM_DOFS_1D*NUM_DOFS_1D, sizeof(dfloat));

  dfloat *cubX = (dfloat*) calloc(NUM_QUAD_1D*NUM_QUAD_1D, sizeof(dfloat));
  dfloat *cubInvX = (dfloat*) calloc(NUM_QUAD_1D*NUM_QUAD_1D, sizeof(dfloat));

  for(int n=0;n<NUM_QUAD_1D;++n){
    cubX[n*NUM_QUAD_1D + n] = 1;
    cubInvX[n*NUM_QUAD_1D + n] = 0.5;

    if(n<NUM_QUAD_1D/2){
      cubX[n*NUM_QUAD_1D + NUM_QUAD_1D-1-n] = -1;
      cubInvX[n*NUM_QUAD_1D + NUM_QUAD_1D-1-n] = +0.5;
    }
    
    if(n>=(NUM_QUAD_1D/2)){
      cubX[n*NUM_QUAD_1D + NUM_QUAD_1D-1-n] = +1;
      cubInvX[n*NUM_QUAD_1D + NUM_QUAD_1D-1-n] = -0.5;
    }
  }

  for(int n=0;n<NUM_DOFS_1D;++n){
    X[n*NUM_DOFS_1D + n] = 1;
    invX[n*NUM_DOFS_1D + n] = 0.5;

    if(n<NUM_DOFS_1D/2){
      X[n*NUM_DOFS_1D + NUM_DOFS_1D-1-n] = 1;
      invX[n*NUM_DOFS_1D + NUM_DOFS_1D-1-n] = -0.5;
    }
    
    if(n>=NUM_DOFS_1D/2){
      X[n*NUM_DOFS_1D + NUM_DOFS_1D-1-n] = -1;
      invX[n*NUM_DOFS_1D + NUM_DOFS_1D-1-n] = 0.5;
    }
  }

  if(NUM_DOFS_1D%2) X[(NUM_DOFS_1D)*(NUM_DOFS_1D)/2] = 1;
  if(NUM_DOFS_1D%2) invX[(NUM_DOFS_1D)*(NUM_DOFS_1D)/2] = 1;
  
  if(NUM_QUAD_1D%2) cubX[(NUM_QUAD_1D)*(NUM_QUAD_1D)/2] = 1;
  if(NUM_QUAD_1D%2) cubInvX[(NUM_QUAD_1D)*(NUM_QUAD_1D)/2] = 1;

  matrixPrint(NUM_DOFS_1D, NUM_DOFS_1D, X, "X");
  matrixPrint(NUM_QUAD_1D, NUM_QUAD_1D, cubX, "cubX");

  
  matrixPrint(NUM_DOFS_1D, NUM_DOFS_1D, invX, "invX");
  matrixPrint(NUM_QUAD_1D, NUM_QUAD_1D, cubInvX, "cubInvX");

  
  dfloat *IinvX = (dfloat*) calloc(NUM_DOFS_1D*NUM_QUAD_1D, sizeof(dfloat));
  dfloat *cubInvXIinvX = (dfloat*) calloc(NUM_DOFS_1D*NUM_QUAD_1D, sizeof(dfloat));

  // post multiply by invX
  for(int i=0;i<NUM_QUAD_1D;++i){
    for(int a=0;a<NUM_DOFS_1D;++a){
      dfloat res = 0;
      for(int n=0;n<NUM_DOFS_1D;++n){
	res += h_DofToQuad[i*NUM_DOFS_1D+n]*invX[n*NUM_DOFS_1D+a];
      }
      IinvX[i*NUM_DOFS_1D+a] = res;
    }
  }

  matrixPrint(NUM_QUAD_1D, NUM_DOFS_1D, IinvX, "IinvX");

  // pre multiply by invX
  for(int i=0;i<NUM_QUAD_1D;++i){
    for(int a=0;a<NUM_DOFS_1D;++a){
      dfloat res = 0;
      for(int n=0;n<NUM_QUAD_1D;++n){
	res += cubInvX[i*NUM_QUAD_1D+n]*IinvX[n*NUM_DOFS_1D + a];
      }
      cubInvXIinvX[i*NUM_DOFS_1D+a] = res;
    }
  }

  matrixPrint(NUM_QUAD_1D, NUM_DOFS_1D, cubInvXIinvX, "cubInvXIinvX");
  
  
  for(int i=0;i<HALF_QUAD_1D;++i){
    for(int a=0;a<HALF_DOFS_1D;++a){

      h_oddDofToQuad[i*HALF_DOFS_1D+a] = cubInvXIinvX[i*NUM_DOFS_1D+a];

      h_evenDofToQuad[i*HALF_DOFS_1D+a] = cubInvXIinvX[(NUM_QUAD_1D-1-i)*NUM_DOFS_1D + NUM_DOFS_1D-1-a];
      
    }
  }

  if((NUM_QUAD_1D%2)) // zero duplicate
    h_evenDofToQuad[HALF_QUAD_1D*HALF_DOFS_1D-1] = 0;

  matrixPrint(HALF_QUAD_1D, HALF_DOFS_1D, h_oddDofToQuad, "h_oddDofToQuad");
  matrixPrint(HALF_QUAD_1D, HALF_DOFS_1D, h_evenDofToQuad, "h_evenDofToQuad");
  
  int NoddDofToQuad = HALF_QUAD_1D*HALF_DOFS_1D;
  int NevenDofToQuad = HALF_QUAD_1D*HALF_DOFS_1D;
  
  cudaMalloc(c_oddDofToQuad, NoddDofToQuad*sizeof(dfloat));
  cudaMalloc(c_evenDofToQuad, NevenDofToQuad*sizeof(dfloat));
  
  cudaMemcpy(*c_oddDofToQuad,  h_oddDofToQuad,  NoddDofToQuad*sizeof(dfloat),  cudaMemcpyHostToDevice);
  cudaMemcpy(*c_evenDofToQuad, h_evenDofToQuad, NoddDofToQuad*sizeof(dfloat), cudaMemcpyHostToDevice);
  
  cudaMemcpyToSymbol(CUDA_SYMBOL(const_oddDofToQuad),  h_oddDofToQuad,  NoddDofToQuad*sizeof(dfloat));
  cudaMemcpyToSymbol(CUDA_SYMBOL(const_evenDofToQuad), h_evenDofToQuad, NoddDofToQuad*sizeof(dfloat));
  cudaMemcpyToSymbol(CUDA_SYMBOL(const_DofToQuad),     h_DofToQuad, NUM_QUAD_1D*NUM_DOFS_1D*sizeof(dfloat));
}


void runBK1Kernel(cudaStream_t stream, int Nq, int cubNq, int numElements,
				 dfloat *c_op,
				 dfloat *c_DofToQuad, dfloat *c_oddDofToQuad, dfloat *c_evenDofToQuad,
				 dfloat *c_solIn, dfloat *c_solOut, int mode){
  
#define BK1Kernel(Nq,cubNq,Nblock)					\
  {									\
    dim3 G((numElements+Nblock-1)/Nblock, 1, 1);			\
    dim3 B(cubNq*cubNq, Nblock, 1);					\
    									\
    if(mode==1)								\
      BK1RegisterKernel<Nq,cubNq,Nblock> <<< G, B, 0, stream >>>( numElements, c_op, c_oddDofToQuad, c_evenDofToQuad, c_solIn, c_solOut); \
    else if(mode==2)							\
      BK1ConstantKernel<Nq,cubNq,Nblock> <<< G, B, 0, stream >>>( numElements, c_op, c_oddDofToQuad, c_evenDofToQuad, c_solIn, c_solOut); \
    else if(mode==3)							\
      BK1SharedKernel<Nq,cubNq,Nblock> <<< G, B, 0, stream >>>( numElements, c_op, c_oddDofToQuad, c_evenDofToQuad, c_solIn, c_solOut); \
    else if(mode==4)							\
      BK1GlobalKernel<Nq,cubNq,Nblock> <<< G, B, 0, stream >>>( numElements, c_op, c_oddDofToQuad, c_evenDofToQuad, c_solIn, c_solOut); \
    else if(mode==5)							\
      BK1MonolithicGlobalKernel<Nq,cubNq,Nblock> <<< G, B, 0, stream >>>( numElements, c_op, c_DofToQuad, c_evenDofToQuad, c_solIn, c_solOut); \
    else if(mode==6)							\
      BK1MonolithicConstantKernel<Nq,cubNq,Nblock> <<< G, B, 0, stream >>>( numElements, c_op, c_DofToQuad, c_evenDofToQuad, c_solIn, c_solOut); \
    else if(mode==7){							\
      dim3 G3(numElements, 1, 1);					\
      dim3 B3(cubNq,cubNq,cubNq);					\
      BK1CubeKernel<Nq,cubNq> <<< G3, B3, 0, stream >>>( numElements, c_op, c_DofToQuad, c_solIn, c_solOut); \
    }									\
  }
  
#define ERR printf("BK1Register with Nq=%d, cubNq=%d not available", Nq, cubNq); exit(-1)

  if(Nq==2){
    switch(cubNq){
    case 2: BK1Kernel(2,2,16); break;
    case 3: BK1Kernel(2,3, 7); break;
    case 4: BK1Kernel(2,4, 4); break;
    case 5: BK1Kernel(2,5, 5); break;
    case 6: BK1Kernel(2,6, 3); break;
    default: ERR;
    }
    return;
  }

  if(Nq==3){
    switch(cubNq){
    case 3: BK1Kernel(3,3,7); break;
    case 4: BK1Kernel(3,4,16); break;
    case 5: BK1Kernel(3,5,5); break;
    case 6: BK1Kernel(3,6,3); break;
    case 7: BK1Kernel(3,7,2); break;
    default: ERR;
    }
    return;
  }

  if(Nq==4){
    switch(cubNq){
    case 4: BK1Kernel(4,4,4); break;
    case 5: BK1Kernel(4,5,5); break;
    case 6: BK1Kernel(4,6,3); break;
    case 7: BK1Kernel(4,7,2); break;
    case 8: BK1Kernel(4,8,1); break;
    default: ERR;
    }
    return;
  }

  if(Nq==5){
    switch(cubNq){
    case 5: BK1Kernel(5,5,5); break;
    case 6: BK1Kernel(5,6,3); break;
    case 7: BK1Kernel(5,7,2); break;
    case 8: BK1Kernel(5,8,1); break;
    case 9: BK1Kernel(5,9,2); break;
    default: ERR;
    }
    return;
  }

  if(Nq==6){
    switch(cubNq){
    case 6:  BK1Kernel(6, 6, 3); break; // Nb=3 best so far
    case 7:  BK1Kernel(6, 7, 2); break;
    case 8:  BK1Kernel(6, 8, 1); break;
    case 9:  BK1Kernel(6, 9, 2); break;
    case 10: BK1Kernel(6,10, 1); break;
    default: ERR;
    }
    return;
  }

  if(Nq==7){
    switch(cubNq){
    case 7:  BK1Kernel(7, 7,2); break;
    case 8:  BK1Kernel(7, 8,1); break;
    case 9:  BK1Kernel(7, 9,2); break;
    case 10: BK1Kernel(7,10,1); break;
    case 11: BK1Kernel(7,11,1); break;

    default: ERR;
    }
    return;
  }

  if(Nq==8){
    switch(cubNq){
    case 8:  BK1Kernel(8, 8,1); break;
    case 9:  BK1Kernel(8, 9,2); break;
    case 10: BK1Kernel(8,10,1); break;
    case 11: BK1Kernel(8,11,1); break;
    case 12: BK1Kernel(8,12,1); break;
    default: ERR;
    }
    return;
  }

  if(Nq==9){
    switch(cubNq){
    case 9:  BK1Kernel(9, 9,1); break;
    case 10: BK1Kernel(9,10,1); break;
    case 11: BK1Kernel(9,11,1); break;
    case 12: BK1Kernel(9,12,1); break;
    case 13: BK1Kernel(9,13,1); break;

    default: ERR;
    }
    return;
  }

  if(Nq==10){
    switch(cubNq){
    case 10: BK1Kernel(10,10,1); break;
    case 11: BK1Kernel(10,11,1); break;
    case 12: BK1Kernel(10,12,1); break;
    case 13: BK1Kernel(10,13,1); break;
    case 14: BK1Kernel(10,14,1); break;
    default: ERR;
    }
    return;
  }

  if(Nq==11){
    switch(cubNq){
    case 11: BK1Kernel(11,11,1); break;
    case 12: BK1Kernel(11,12,1); break;
    case 13: BK1Kernel(11,13,1); break;
    case 14: BK1Kernel(11,14,1); break;
    case 15: BK1Kernel(11,15,1); break;

    default: ERR;
    }
    return;
  }
  
  if(Nq==12){
    switch(cubNq){
    case 12: BK1Kernel(12,12,1); break;
    case 13: BK1Kernel(12,13,1); break;
    case 14: BK1Kernel(12,14,1); break;
    case 15: BK1Kernel(12,15,1); break;
      //    case 16: BK1Kernel(12,16,1); break;
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


int main(int argc, char **argv){

  cudaSetDevice(0);
  
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  if(argc!=5){
    printf("Usage: ./BK1VT Nq cubNq numElements mode \n");
    exit(-1);
  }

  // read number of elements
  int        Nq = atoi(argv[1]);
  int     cubNq = atoi(argv[2]);
  int numElements = atoi(argv[3]);
  int        mode = atoi(argv[4]);

  if(mode==0 || mode>7) {
    printf("Exiting: mode %d not supported\n", mode);
  }
  
  printf("Running: NUM_DOFS_1D=%d, NUM_QUAD_1D=%d, numElements=%d, mode=%d\n", Nq, cubNq, numElements, mode);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);	

  int Ntests = 50;
  
  // do nothing kernel test
  dfloat nothingElapsed = nothingTest(stream, Ntests);
  nothingElapsed = nothingTest(stream, Ntests);

  int   Np = Nq*Nq*Nq;
  int   cubNp = cubNq*cubNq*cubNq;

  int halfNq = ((Nq+1)/2);
  int halfCubNq = ((cubNq+1)/2);

  int    Ntotal = numElements*Np;
  int cubNtotal = numElements*cubNp;

  // bandwidth test
  // total number of bytes

  double estimatedActualDeviceBandwidth = bandwidthTest(stream, Ntests, (Ntotal*2+cubNtotal)*sizeof(dfloat));
  
  dfloat *h_op,      *c_op;
  dfloat *h_solOut,       *c_solOut;
  dfloat *h_solIn,        *c_solIn;
  dfloat *h_DofToQuad,    *c_DofToQuad;
  dfloat *h_oddDofToQuad, *c_oddDofToQuad;
  dfloat *h_evenDofToQuad, *c_evenDofToQuad;


  // float fields
  randAlloc(cubNtotal*p_Nvgeo, &h_op, &c_op);

  for(int e=0;e<numElements;++e){
    for(int n=0;n<cubNp;++n){
      h_op[e*cubNp+n] = drand48();
    }
  }
  
  cudaMemcpy(c_op, h_op, p_Nvgeo*numElements*cubNp*sizeof(dfloat), cudaMemcpyHostToDevice);
  
  randAlloc(Ntotal, &h_solIn, &c_solIn);
  randAlloc(Ntotal, &h_solOut, &c_solOut);
  
  randAlloc(Nq*cubNq, &h_DofToQuad, &c_DofToQuad);
  randAlloc(halfNq*halfCubNq, &h_oddDofToQuad, &c_oddDofToQuad);
  randAlloc(halfNq*halfCubNq, &h_evenDofToQuad, &c_evenDofToQuad);

  // build interpolation matrix
  dfloat *r, *w, *cubr, *cubw;
  meshJacobiGL(0,0,Nq-1, &r, &w);
  meshJacobiGQ(0,0,cubNq-1, &cubr, &cubw);
  meshInterpolationMatrix1D(Nq-1, Nq, r, cubNq, cubr, &h_DofToQuad);
  cudaMemcpy(c_DofToQuad, h_DofToQuad, cubNq*Nq*sizeof(dfloat), cudaMemcpyHostToDevice);
  
  matrixPrint(cubNq, Nq, h_DofToQuad, "DofToQuad");

  // create Odd-even packed storage for I and transpose(I) and push to constant memory
  buildInterpMatrices (Nq,cubNq, h_DofToQuad, h_oddDofToQuad, h_evenDofToQuad,
		       &c_oddDofToQuad, &c_evenDofToQuad);


  // KERNEL GRID
  float elapsed;
  
  // warm up call
  runBK1Kernel (stream, Nq, cubNq, numElements, c_op, c_DofToQuad, c_oddDofToQuad, c_evenDofToQuad, c_solIn, c_solOut, mode);

  cudaDeviceSynchronize();

  {
    cudaEventRecord(start, stream);
    
    for(int test=0;test<Ntests;++test)
      runBK1Kernel (stream, Nq, cubNq, numElements, c_op, c_DofToQuad, c_oddDofToQuad, c_evenDofToQuad, c_solIn, c_solOut, mode);

    cudaEventRecord(end, stream);

    cudaDeviceSynchronize();
    
    cudaEventElapsedTime(&elapsed, start, end);
    elapsed /= 1000.;
    elapsed /= (double) Ntests;

    // estimate bandwidth (assuming all data moved to/from device memory)
    int bytesMoved = (2*Np+cubNp)*sizeof(dfloat); // x, Mx, opa   
    double bw = (bytesMoved*numElements/elapsed)/1.e9;

    double estFlops =
      numElements*(( Nq*Nq*(halfNq*2 + halfCubNq*(halfNq*4 +2)) +
		     Nq*cubNq*(halfNq*2 + halfCubNq*(halfNq*4 + 2)) + 
		     cubNq*cubNq*(halfNq*2 + halfCubNq*(halfNq*4+6) + halfNq*(halfCubNq*4+2)) + 
		     Nq*cubNq*(halfNq*2+halfNq*(halfCubNq*4 + 2)) +
		     Nq*Nq*(halfCubNq*2 + halfNq*(halfCubNq*4 + 2)))/elapsed)/1.e9;

    double effectiveFlops =
      numElements*(2*( Nq*Nq*Nq*cubNq*2 + Nq*Nq*cubNq*cubNq*2 + Nq*cubNq*cubNq*cubNq*2)/elapsed)/1.e9;
    
    printf("%2d %2d %8d %8d %e %e %e %e %e %e %e %d %%%% [BK1: NUM_DOFS_1D, NUM_QUAD_1D, numElements, Ndofs,"
	   " elapsed, dofsPerSecond, nothingElapsed, BW in GB/s, estimated peak Device BW, est. GFLOPS/s, oddeven GFLOPS/s, mode]\n",
	   Nq, cubNq, numElements, Np*numElements, elapsed, numElements*(Np/elapsed), nothingElapsed, bw, estimatedActualDeviceBandwidth, estFlops, effectiveFlops, mode);
  }

  // check output is correct
  meshReferenceBK1(Nq, cubNq, numElements, h_op, h_DofToQuad, h_solIn, h_solOut);

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

  printf("NUM_DOFS_1D=%02d, NUM_QUAD_1D=%02d || Mq_{host} - Mq_{device} ||_linf = %lg\n", Nq, cubNq, maxDiff);

  cudaEventDestroy(start);
  cudaEventDestroy(end);	
  
  return 0;

}
