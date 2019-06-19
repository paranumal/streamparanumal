/*

See LICENSE file.

*/

#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

#define dfloat double

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
// 1 to use HIP 10.0 stream recording
// 0 to use traditional enqueing of kernels
#define USE_GRAPH 0

#define MAX_DOFS_1D 14
#define MAX_HALF_DOFS_1D 7


#define HALF_DOFS_1D ((NUM_DOFS_1D+1)/2)

#define NUM_DOFS_2D (NUM_DOFS_1D*NUM_DOFS_1D)
#define NUM_DOFS_3D (NUM_DOFS_1D*NUM_DOFS_1D*NUM_DOFS_1D)

#define p_Nop 7

#define p_G00ID 0
#define p_G01ID 1
#define p_G02ID 2
#define p_G11ID 3
#define p_G12ID 4
#define p_G22ID 5
#define p_GWJID 6


__constant__ dfloat const_DofToDofD[MAX_DOFS_1D*MAX_DOFS_1D];
__constant__ dfloat const_oddDofToDofD[MAX_HALF_DOFS_1D*MAX_HALF_DOFS_1D];
__constant__ dfloat const_evenDofToDofD[MAX_HALF_DOFS_1D*MAX_HALF_DOFS_1D];

void randAlloc(int N, dfloat **h_a, dfloat **c_a){

  *h_a = (dfloat*) calloc(N, sizeof(dfloat));

  for(int n=0;n<N;++n)
    h_a[0][n] = drand48();

  hipMalloc(c_a, N*sizeof(dfloat));

  hipMemcpy(c_a[0], h_a[0], N*sizeof(dfloat), hipMemcpyHostToDevice);

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
		 dfloat s_p[p_Nblock][NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D],
		 dfloat * __restrict__ r_Ap){
  
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
#pragma nounroll 
  for(int k = 0; k < NUM_DOFS_1D; k++) {

    dfloat G00 = 0, G01 =0, G02 =0, G11 =0, G12 =0, G22 =0, GWJ =0;
    
    // prefetch geometric factors
    const int gbase = element*p_Nop*NUM_DOFS_3D + ijkN(i,j,k,NUM_DOFS_1D);

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
      pr += DofToDofD[im]*s_p[blk][k][j][m];
      ps += DofToDofD[jm]*s_p[blk][k][m][i];
      pt += DofToDofD[km]*s_p[blk][m][j][i];
    }

    __syncthreads();
    
    s_Gpr[blk][j][i] = (G00*pr + G01*ps + G02*pt);
    s_Gps[blk][j][i] = (G01*pr + G11*ps + G12*pt);
    
    dfloat Gpt = (G02*pr + G12*ps + G22*pt);
    
    dfloat Apk = GWJ*lambda*s_p[blk][k][j][i];
    
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
  __shared__ dfloat s_p[p_Nblock][NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D];
  
  dfloat r_Ap[NUM_DOFS_1D];

  const unsigned int t = threadIdx.x;
  const int blk = threadIdx.y;
  
  const int element = blockIdx.x*p_Nblock + blk;
  
  const unsigned int a = t%NUM_DOFS_1D;
  const unsigned int b = t/NUM_DOFS_1D;

  if(blk==0)
    s_DofToDofD[t] = DofToDofD[t];
  
  if(element < numElements){
    for(int c=0;c<NUM_DOFS_1D;++c){
      
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      
      s_p[blk][c][b][a] = solIn[id];
    }
  }
  
  __syncthreads();
  
  BK5Device  <NUM_DOFS_1D, p_Nblock>
    (numElements, element, lambda, op, DofToDofD, const_oddDofToDofD, const_evenDofToDofD, s_p, r_Ap);
  
  if(element<numElements){
#pragma unroll
    for(int c=0;c<NUM_DOFS_1D;++c){
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      solOut[id] = r_Ap[c];
    }
  }
}

template <int NUM_DOFS_1D>
__global__ void BK5ImportKernel(const int numElements,
				const dfloat  lambda,			     
				const  dfloat  * __restrict__ op,
				const  dfloat  * __restrict__ D,
				const  dfloat  * __restrict__ q,
				dfloat  *  __restrict__ Aq){
  
  
  __shared__ dfloat  s_D[NUM_DOFS_1D][NUM_DOFS_1D];
  __shared__ dfloat  s_q[NUM_DOFS_1D][NUM_DOFS_1D];
  
  __shared__ dfloat  s_Gqr[NUM_DOFS_1D][NUM_DOFS_1D];
  __shared__ dfloat  s_Gqs[NUM_DOFS_1D][NUM_DOFS_1D];
  
  dfloat  r_qt, r_Gqt, r_Auk;
  dfloat  r_q[NUM_DOFS_1D]; // register array to hold u(i,j,0:N) private to thread
  dfloat  r_Aq[NUM_DOFS_1D];// array for results Au(i,j,0:N)
  
  dfloat  r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;

  int e = blockIdx.x;
  const unsigned int t = threadIdx.x;
  
  int i=t%NUM_DOFS_1D;
  int j=t/NUM_DOFS_1D;

  //load D into local memory
  // s_D[i][j] = d \phi_i at node j
  s_D[j][i] = D[NUM_DOFS_1D*j+i]; // D is column major
  
  // load pencil of u into register
  const int base = i + j*NUM_DOFS_1D + e*NUM_DOFS_3D;
  for(int k = 0; k < NUM_DOFS_1D; k++) {
    r_q[k] = q[base + k*NUM_DOFS_1D*NUM_DOFS_1D]; // prefetch operation
    r_Aq[k] = 0.f; // zero the accumulator
  }

  //#pragma nounroll NUM_DOFS_1D
  for(int k = 0;k < NUM_DOFS_1D; k++){
	
    // prefetch geometric factors
    const int gbase = e*p_Nop*NUM_DOFS_3D + k*NUM_DOFS_1D*NUM_DOFS_1D + j*NUM_DOFS_1D + i;
    
    r_G00 = op[gbase+p_G00ID*NUM_DOFS_3D];
    r_G01 = op[gbase+p_G01ID*NUM_DOFS_3D];
    r_G02 = op[gbase+p_G02ID*NUM_DOFS_3D];
    
    r_G11 = op[gbase+p_G11ID*NUM_DOFS_3D];
    r_G12 = op[gbase+p_G12ID*NUM_DOFS_3D];
    r_G22 = op[gbase+p_G22ID*NUM_DOFS_3D];
    
    r_GwJ = op[gbase+p_GWJID*NUM_DOFS_3D];

    __syncthreads();

    // share u(:,:,k)
    s_q[j][i] = r_q[k];
    
    r_qt = 0;

    //#pragma unroll NUM_DOFS_1D
    for(int m = 0; m < NUM_DOFS_1D; m++) {
      r_qt += s_D[k][m]*r_q[m];
    }

    __syncthreads();

    dfloat  qr = 0.f;
    dfloat  qs = 0.f;
    
    //#pragma unroll NUM_DOFS_1D
    for(int m = 0; m < NUM_DOFS_1D; m++) {
      qr += s_D[i][m]*s_q[j][m];
      qs += s_D[j][m]*s_q[m][i];
    }

    s_Gqr[j][i] = (r_G00*qr + r_G01*qs + r_G02*r_qt);
    s_Gqs[j][i] = (r_G01*qr + r_G11*qs + r_G12*r_qt);
    
    // put this here for a performance bump
    r_Gqt = (r_G02*qr + r_G12*qs + r_G22*r_qt);
    r_Auk = r_GwJ*lambda*r_q[k];

    __syncthreads();

    //#pragma unroll NUM_DOFS_1D
    for(int m = 0; m < NUM_DOFS_1D; m++){
      r_Auk   += s_D[m][j]*s_Gqs[m][i];
      r_Aq[m] += s_D[k][m]*r_Gqt; // DT(m,k)*ut(i,j,k,e)
      r_Auk   += s_D[m][i]*s_Gqr[j][m];
    }
    
    r_Aq[k] += r_Auk;
  }
  
  // write out
  //#pragma unroll NUM_DOFS_1D
  for(int k = 0; k < NUM_DOFS_1D; k++){
    const int id = e*NUM_DOFS_3D +k*NUM_DOFS_1D*NUM_DOFS_1D+ j*NUM_DOFS_1D + i;
    Aq[id] = r_Aq[k];
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

__global__ void BK5SharedKernel(const int numElements,
				const dfloat  lambda,			     
				const  dfloat  * __restrict__ op,
				const  dfloat  * __restrict__ D,
				const  dfloat  * __restrict__ q,
				dfloat  *  __restrict__ Aq){
  
  
  __shared__ dfloat  s_D[NUM_DOFS_1D][NUM_DOFS_1D];
  __shared__ dfloat  s_q[NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D];
  __shared__ dfloat  s_Aq[NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D];
  
  __shared__ dfloat  s_Gqr[NUM_DOFS_1D][NUM_DOFS_1D];
  __shared__ dfloat  s_Gqs[NUM_DOFS_1D][NUM_DOFS_1D];
  
  //  dfloat  r_Aq[NUM_DOFS_1D];// array for results Au(i,j,0:N)
  
  int e = blockIdx.x;
  const unsigned int t = threadIdx.x;
  
  int i=t%NUM_DOFS_1D;
  int j=t/NUM_DOFS_1D;
>>>>>>> c2d2fcc039cc0f732e002e214a3ef4719db4230a

  //load D into local memory
  // s_D[i][j] = d \phi_i at node j
  s_D[j][i] = D[NUM_DOFS_1D*j+i]; // D is column major
  
  // load pencil of u into register
  const int base = i + j*NUM_DOFS_1D + e*NUM_DOFS_3D;
  for(int k = 0; k < NUM_DOFS_1D; k++) {
    dfloat qkji = q[base + k*NUM_DOFS_1D*NUM_DOFS_1D]; // prefetch operation

    // prefetch geometric factors
    const int gbase = e*p_Nop*NUM_DOFS_3D + k*NUM_DOFS_1D*NUM_DOFS_1D + j*NUM_DOFS_1D + i;
    
    dfloat r_GwJ = op[gbase+p_GWJID*NUM_DOFS_3D];
    
    s_q[k][j][i] = qkji;
    s_Aq[k][j][i] = r_GwJ*lambda*qkji;
  }

  __syncthreads();

#if 0
  dfloat r_Di[NUM_DOFS_1D];
  dfloat r_Dj[NUM_DOFS_1D];
  for(int m=0;m<NUM_DOFS_1D;++m){
    r_Di[m] = s_D[i][m];
    r_Dj[m] = s_D[j][m];
  }
#endif

  // NUM_DOFS_1D
  //#pragma nounroll 
  for(int k = 0;k < NUM_DOFS_1D; k++){

    __syncthreads();

    // prefetch geometric factors
    const int gbase = e*p_Nop*NUM_DOFS_3D + k*NUM_DOFS_1D*NUM_DOFS_1D + j*NUM_DOFS_1D + i;

    dfloat r_Gqr = 0, r_Gqs = 0, r_Gqt = 0;

    {
      dfloat r_G00 = op[gbase+p_G00ID*NUM_DOFS_3D];
      dfloat r_G01 = op[gbase+p_G01ID*NUM_DOFS_3D];
      dfloat r_G02 = op[gbase+p_G02ID*NUM_DOFS_3D];

      dfloat  qr = 0.f;
      for(int m = 0; m < NUM_DOFS_1D; m++){
	qr += s_D[i][m]*s_q[k][j][m];
	//	qr += r_Di[m]*s_q[k][j][m];
      }

      r_Gqr += r_G00*qr;
      r_Gqs += r_G01*qr;
      r_Gqt += r_G02*qr;
      
      dfloat r_G11 = op[gbase+p_G11ID*NUM_DOFS_3D];
      dfloat r_G12 = op[gbase+p_G12ID*NUM_DOFS_3D];

      dfloat  qs = 0.f;
      for(int m = 0; m < NUM_DOFS_1D; m++) {
	qs += s_D[j][m]*s_q[k][m][i];
	//	qs += r_Dj[m]*s_q[k][m][i];
      }
      
      r_Gqr += r_G01*qs;
      r_Gqs += r_G11*qs;
      r_Gqt += r_G12*qs;
      
      dfloat r_G22 = op[gbase+p_G22ID*NUM_DOFS_3D];

      dfloat qt = 0;
      for(int m = 0; m < NUM_DOFS_1D; m++) {
	qt += s_D[k][m]*s_q[m][j][i];
      }
      
      r_Gqr += r_G02*qt;
      r_Gqs += r_G12*qt;
      r_Gqt += r_G22*qt;
      
      s_Gqr[j][i] = r_Gqr;
      s_Gqs[j][i] = r_Gqs;
    }
    
    // put this here for a bump
    for(int m = 0; m < NUM_DOFS_1D; m++)
      s_Aq[m][j][i] += s_D[k][m]*r_Gqt;

    // constant memory is slow for some reason
    //s_Aq[m][j][i] += const_DofToDofD[k*NUM_DOFS_1D+m]*r_Gqt; 
    
    __syncthreads();
    
    dfloat r_Auks = 0;
    for(int m = 0; m < NUM_DOFS_1D; m++){
      r_Auks   += s_D[m][j]*s_Gqs[m][i];
    }
    
    dfloat r_Aukr = 0;
    for(int m = 0; m < NUM_DOFS_1D; m++){
      r_Aukr   += s_D[m][i]*s_Gqr[j][m];
    }
    
    s_Aq[k][j][i] += r_Aukr+r_Auks;
  }
  
  // write out
  //#pragma unroll NUM_DOFS_1D
  for(int k = 0; k < NUM_DOFS_1D; k++){
    const int id = e*NUM_DOFS_3D +k*NUM_DOFS_1D*NUM_DOFS_1D+ j*NUM_DOFS_1D + i;
    Aq[id] = s_Aq[k][j][i]; // r_Aq[k];
  }
}

template <int NUM_DOFS_1D, int p_Nblock>
__global__ void BK5BlockedSharedKernel(const int numElements,
				       const dfloat  lambda,			     
				       const  dfloat  * __restrict__ op,
				       const  dfloat  * __restrict__ D,
				       const  dfloat  * __restrict__ q,
				       dfloat  *  __restrict__ Aq){
  
  
  __shared__ dfloat  s_D[NUM_DOFS_1D][NUM_DOFS_1D];

  __shared__ dfloat  s_q[p_Nblock][NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D];
  __shared__ dfloat  s_Aq[p_Nblock][NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D];

  __shared__ dfloat  s_Gqr[p_Nblock][NUM_DOFS_1D][NUM_DOFS_1D];
  __shared__ dfloat  s_Gqs[p_Nblock][NUM_DOFS_1D][NUM_DOFS_1D];
  
  //  dfloat  r_Aq[NUM_DOFS_1D];// array for results Au(i,j,0:N)
  const unsigned int t = threadIdx.x;
  int blk = threadIdx.y;

  int element = blk + blockIdx.x*p_Nblock;
  
  int i=t%NUM_DOFS_1D;
  int j=t/NUM_DOFS_1D;

  //load D into local memory
  // s_D[i][j] = d \phi_i at node j
  if(blk==0)
    s_D[j][i] = D[NUM_DOFS_1D*j+i]; // D is column major
  
  // load pencil of u into register
  const int base = i + j*NUM_DOFS_1D + element*NUM_DOFS_3D;
  if(element<numElements)
    for(int k = 0; k < NUM_DOFS_1D; k++) {
      dfloat qkji = q[base + k*NUM_DOFS_1D*NUM_DOFS_1D]; // prefetch operation
      
      // prefetch geometric factors
      const int gbase = element*p_Nop*NUM_DOFS_3D + k*NUM_DOFS_1D*NUM_DOFS_1D + j*NUM_DOFS_1D + i;
      
      dfloat r_GwJ = op[gbase+p_GWJID*NUM_DOFS_3D];
      
      s_q[blk][k][j][i] = qkji;
      s_Aq[blk][k][j][i] = r_GwJ*lambda*qkji;
    }

  // NUM_DOFS_1D
  //#pragma nounroll 
  for(int k = 0;k < NUM_DOFS_1D; k++){
    
    __syncthreads();

    // prefetch geometric factors
    const int gbase = element*p_Nop*NUM_DOFS_3D + k*NUM_DOFS_1D*NUM_DOFS_1D + j*NUM_DOFS_1D + i;

    dfloat r_Gqr = 0, r_Gqs = 0, r_Gqt = 0;

    if(element<numElements){
      
      dfloat r_G00 = op[gbase+p_G00ID*NUM_DOFS_3D];
      dfloat r_G01 = op[gbase+p_G01ID*NUM_DOFS_3D];
      dfloat r_G02 = op[gbase+p_G02ID*NUM_DOFS_3D];

      dfloat  qr = 0.f;
      for(int m = 0; m < NUM_DOFS_1D; m++){
	qr += s_D[i][m]*s_q[blk][k][j][m];
      }

      r_Gqr += r_G00*qr;
      r_Gqs += r_G01*qr;
      r_Gqt += r_G02*qr;
      
      dfloat r_G11 = op[gbase+p_G11ID*NUM_DOFS_3D];
      dfloat r_G12 = op[gbase+p_G12ID*NUM_DOFS_3D];

      dfloat  qs = 0.f;
      for(int m = 0; m < NUM_DOFS_1D; m++) {
	qs += s_D[j][m]*s_q[blk][k][m][i];
      }
      
      r_Gqr += r_G01*qs;
      r_Gqs += r_G11*qs;
      r_Gqt += r_G12*qs;
      
      dfloat r_G22 = op[gbase+p_G22ID*NUM_DOFS_3D];

      dfloat qt = 0;
      for(int m = 0; m < NUM_DOFS_1D; m++) {
	qt += s_D[k][m]*s_q[blk][m][j][i];
      }
      
      r_Gqr += r_G02*qt;
      r_Gqs += r_G12*qt;
      r_Gqt += r_G22*qt;
      
      s_Gqr[blk][j][i] = r_Gqr;
      s_Gqs[blk][j][i] = r_Gqs;
    }
    
    // put this here for a bump
    for(int m = 0; m < NUM_DOFS_1D; m++)
      s_Aq[blk][m][j][i] += s_D[k][m]*r_Gqt;

    // constant memory is slow for some reason
    //s_Aq[m][j][i] += const_DofToDofD[k*NUM_DOFS_1D+m]*r_Gqt; 
    
    __syncthreads();
    
    dfloat r_Auks = 0;
    for(int m = 0; m < NUM_DOFS_1D; m++){
      r_Auks   += s_D[m][j]*s_Gqs[blk][m][i];
    }
    
    dfloat r_Aukr = 0;
    for(int m = 0; m < NUM_DOFS_1D; m++){
      r_Aukr   += s_D[m][i]*s_Gqr[blk][j][m];
    }
    
    s_Aq[blk][k][j][i] += r_Aukr+r_Auks;
  }
  
  // write out
  //#pragma unroll NUM_DOFS_1D
  	      
  if(element<numElements)
  for(int k = 0; k < NUM_DOFS_1D; k++){
    const int id = element*NUM_DOFS_3D +k*NUM_DOFS_1D*NUM_DOFS_1D+ j*NUM_DOFS_1D + i;
    Aq[id] = s_Aq[blk][k][j][i]; // r_Aq[k];
  }
}



void BK5Host(int NUM_DOFS_1D, int numElements, dfloat lambda,
	     const dfloat * __restrict__ op,
	     const dfloat * __restrict__ DofToDofD,
	     const dfloat * q,
	     dfloat *lapqout){

  
  dfloat Gqr[NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D];
  dfloat Gqs[NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D];
  dfloat Gqt[NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D];
  
  for(int element=0;element<numElements;++element){
    
    for(int k=0;k<NUM_DOFS_1D;++k){
      for(int j=0;j<NUM_DOFS_1D;++j){
	for(int i=0;i<NUM_DOFS_1D;++i){
	  
	  dfloat qr = 0;
	  dfloat qs = 0;
	  dfloat qt = 0;
	  
	  for(int n=0;n<NUM_DOFS_1D;++n){
	    int in = ijN(n,i,NUM_DOFS_1D);
	    int jn = ijN(n,j,NUM_DOFS_1D);
	    int kn = ijN(n,k,NUM_DOFS_1D);
	    
	    int kjn = ijklN(n,j,k,element,NUM_DOFS_1D);
	    int kni = ijklN(i,n,k,element,NUM_DOFS_1D);
	    int nji = ijklN(i,j,n,element,NUM_DOFS_1D);
	    
	    qr += DofToDofD[in]*q[kjn];
	    qs += DofToDofD[jn]*q[kni];
	    qt += DofToDofD[kn]*q[nji];	  
	  }
	  
	  const int gbase = element*p_Nop*NUM_DOFS_3D + ijkN(i,j,k,NUM_DOFS_1D);
	  
	  dfloat G00 = op[gbase+p_G00ID*NUM_DOFS_3D];
	  dfloat G01 = op[gbase+p_G01ID*NUM_DOFS_3D];
	  dfloat G02 = op[gbase+p_G02ID*NUM_DOFS_3D];
	  dfloat G11 = op[gbase+p_G11ID*NUM_DOFS_3D];
	  dfloat G12 = op[gbase+p_G12ID*NUM_DOFS_3D];
	  dfloat G22 = op[gbase+p_G22ID*NUM_DOFS_3D];
	  
	  Gqr[k][j][i] = (G00*qr + G01*qs + G02*qt);
	  Gqs[k][j][i] = (G01*qr + G11*qs + G12*qt);
	  Gqt[k][j][i] = (G02*qr + G12*qs + G22*qt);
	}
      }
    }
    
    
    for(int k=0;k<NUM_DOFS_1D;++k){
      for(int j=0;j<NUM_DOFS_1D;++j){
	for(int i=0;i<NUM_DOFS_1D;++i){
	  
	  int kji = ijklN(i,j,k,element,NUM_DOFS_1D);
	  
	  const int gbase = element*p_Nop*NUM_DOFS_3D + ijkN(i,j,k,NUM_DOFS_1D);
	  
	  dfloat GWJ = op[gbase+p_GWJID*NUM_DOFS_3D];
	  dfloat lapq = lambda*GWJ*q[kji];
	  
	  for(int n=0;n<NUM_DOFS_1D;++n){
	    int ni = ijN(i,n,NUM_DOFS_1D);
	    int nj = ijN(j,n,NUM_DOFS_1D);
	    int nk = ijN(k,n,NUM_DOFS_1D);
	    
	    lapq += DofToDofD[ni]*Gqr[k][j][n];
	    lapq += DofToDofD[nj]*Gqs[k][n][i];
	    lapq += DofToDofD[nk]*Gqt[n][j][i];	  
	  }
	  
	  lapqout[kji] = lapq;
	}
      }
    }
  }
}


double bandwidthTest(hipStream_t stream, int Ntests, size_t bwNtotal){

  hipEvent_t start, end;
  hipEventCreate(&start);
  hipEventCreate(&end);	
  
  dfloat *h_bwTest1, *c_bwTest1;
  dfloat *h_bwTest2, *c_bwTest2;
  
  randAlloc(bwNtotal/2, &h_bwTest1, &c_bwTest1);
  randAlloc(bwNtotal/2, &h_bwTest2, &c_bwTest2);
  
  hipDeviceSynchronize();
  hipEventRecord(start, stream);
  
  for(int test=0;test<Ntests/2;++test){
    hipMemcpy(c_bwTest2, c_bwTest1, (bwNtotal/2)*sizeof(dfloat), hipMemcpyDeviceToDevice);
    hipMemcpy(c_bwTest1, c_bwTest2, (bwNtotal/2)*sizeof(dfloat), hipMemcpyDeviceToDevice);
  }
  
  hipEventRecord(end, stream);
  hipEventSynchronize(end);
  hipDeviceSynchronize();

  float elapsed;
  hipEventElapsedTime(&elapsed, start, end);
  elapsed /= 1000.; // convert to s
  elapsed /= (double) Ntests;
  
  double estimatedActualDeviceBandwidth = (bwNtotal*sizeof(dfloat)/elapsed)/1.e9;
  
  hipFree(c_bwTest1);
  hipFree(c_bwTest2);
  
  free(h_bwTest1);
  free(h_bwTest2);
  
  hipEventDestroy(start);
  hipEventDestroy(end);	
  
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
  
  hipMalloc(c_oddOP, NoddOP*sizeof(dfloat));
  hipMalloc(c_evenOP, NevenOP*sizeof(dfloat));
  
  hipMemcpy(*c_oddOP,  oddOP,  NoddOP*sizeof(dfloat),  hipMemcpyHostToDevice);
  hipMemcpy(*c_evenOP, evenOP, NoddOP*sizeof(dfloat), hipMemcpyHostToDevice);

  hipMemcpy(*c_OP, h_OP,  NUM_COLS_OP*NUM_ROWS_OP*sizeof(dfloat),  hipMemcpyHostToDevice);

  matrixPrint(NUM_COLS_OP, NUM_COLS_OP, X, "X");
  matrixPrint(NUM_ROWS_OP, NUM_ROWS_OP, cubX, "cubX");

  
  matrixPrint(NUM_COLS_OP, NUM_COLS_OP, invX, "invX");
  matrixPrint(NUM_ROWS_OP, NUM_ROWS_OP, cubInvX, "cubInvX");


  
}


void runBK5Kernel(hipStream_t stream, int Nq, int numElements, dfloat lambda,
		  dfloat *c_op,
		  dfloat *c_DofToDofD, dfloat *c_oddDofToDofD, dfloat *c_evenDofToDofD,
		  dfloat *c_solIn, dfloat *c_solOut, int mode){
  
#define BK5Kernel(Nq,Nblock)						\
  {									\
    if(mode==0){							\
      dim3 G((numElements+Nblock-1)/Nblock, 1, 1);			\
      dim3 B(Nq*Nq, Nblock, 1);						\
      hipLaunchKernelGGL(BK5ConstantKernel<Nq,Nblock>, G, B, 0, stream,	\
			 numElements, lambda, c_op, c_DofToDofD, c_oddDofToDofD,c_evenDofToDofD, c_solIn, c_solOut); \
    }else if(mode==1){							\
      dim3 G(numElements, 1, 1);					\
      dim3 B(Nq*Nq, 1, 1);						\
      hipLaunchKernelGGL(BK5SharedKernel<Nq>, G, B, 0, stream,		\
			 numElements, lambda, c_op, c_DofToDofD, c_solIn, c_solOut); \
    }									\
    else if(mode==2){							\
      dim3 G((numElements+Nblock-1)/Nblock, 1, 1);			\
      dim3 B(Nq*Nq, Nblock, 1);						\
      hipLaunchKernelGGL(BK5BlockedSharedKernel<Nq,Nblock>, G, B, 0, stream, \
			 numElements, lambda, c_op, c_DofToDofD, c_solIn, c_solOut); \
    }else if(mode==2){							\
      dim3 G(numElements, 1, 1);					\
      dim3 B(Nq,Nq, Nq);						\
      hipLaunchKernelGGL(BK5CubeKernel<Nq>, G, B, 0, stream,		\
			 numElements, lambda, c_op, c_DofToDofD, c_solIn, c_solOut); \
    }									\

    }
  }
  
  //      hipLaunchKernelGGL(BK5ImportKernel<Nq>, G, B, 0, stream,	\
  //			 numElements, lambda, c_op, c_DofToDofD, c_solIn, c_solOut); \

  
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
    //    BK5Kernel(5,5);
    BK5Kernel(5,2);
    return;
  }

  if(Nq==6){
    BK5Kernel(6,3);
    return;
  }

  if(mode==2)
    mode = 1;
  
  if(Nq==7){
    BK5Kernel(7,1);
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


dfloat nothingTest(hipStream_t stream, int Ntests){

  hipEvent_t start, end;
  hipEventCreate(&start);
  hipEventCreate(&end);	

  hipDeviceSynchronize();
  
  float nothingElapsed = 0;
  {
    
    // time kernel that does nothing
    
#if USE_GRAPH==1
    // hip stream capture sequence for nothingKernel
    hipGraph_t nothingGraph;
    
    hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
    
    for(int test=0;test<Ntests;++test){
      hipLaunchKernelGGL(nothingKernel,  1, 1, 0, stream);
    }
    
    hipStreamEndCapture(stream, &nothingGraph);
    
    // time graph sequence for nothing
    hipGraphExec_t nothingInstance;
    hipGraphInstantiate(&nothingInstance, nothingGraph, NULL, NULL, 0);
    
    hipEventRecord(start, stream);
    
    hipGraphLaunch(nothingInstance, stream);
    
    hipEventRecord(end, stream);
#else
    
    hipEventRecord(start, stream);
    
    for(int test=0;test<Ntests;++test)
      hipLaunchKernelGGL(nothingKernel, 1, 1, 0, stream);
    
    hipEventRecord(end, stream);
    
#endif
    
    hipDeviceSynchronize();
    
    hipEventElapsedTime(&nothingElapsed, start, end);
    nothingElapsed /= 1000.;
    nothingElapsed /= (double) Ntests;
    
  }

  return nothingElapsed;
}


int main(int argc, char **argv){

  hipStream_t stream;
  hipStreamCreate(&stream);
  
  if(argc!=4){
    printf("Usage: ./massMatrixMultiplyVT Nq numElements mode\n");
    exit(-1);
  }

  // read number of elements
  int          Nq = atoi(argv[1]);
  int numElements = atoi(argv[2]);
  int        mode = atoi(argv[3]);
  
  dfloat lambda = 0;
  
  printf("Running: NUM_DOFS_1D=%d, numElements=%d, mode=%d\n", Nq, numElements, mode);

  int   Np = Nq*Nq*Nq;
  int halfNq = ((Nq+1)/2);

  int    Ntotal = numElements*Np;

  int Ntests = 100;
  
  double estimatedActualDeviceBandwidth = bandwidthTest(stream, Ntests, (Ntotal*2+7*Ntotal)*sizeof(dfloat));
  
  dfloat *h_op,      *c_op;
  dfloat *h_solOut,       *c_solOut;
  dfloat *h_solIn,        *c_solIn;

  dfloat *h_DofToDofD,    *c_DofToDofD;
  dfloat *c_oddDofToDofD, *c_evenDofToDofD;

  // float fields
  randAlloc(Ntotal*p_Nop, &h_op, &c_op);
  
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

  hipMemcpyToSymbol(HIP_SYMBOL(const_DofToDofD),     c_DofToDofD,     Nq*Nq*sizeof(dfloat), 0, hipMemcpyDeviceToDevice);
  hipMemcpyToSymbol(HIP_SYMBOL(const_oddDofToDofD),  c_oddDofToDofD,  halfNq*halfNq*sizeof(dfloat), 0, hipMemcpyDeviceToDevice);
  hipMemcpyToSymbol(HIP_SYMBOL(const_evenDofToDofD), c_evenDofToDofD, halfNq*halfNq*sizeof(dfloat), 0, hipMemcpyDeviceToDevice);
  
  hipEvent_t start, end;
  hipEventCreate(&start);
  hipEventCreate(&end);	

  // KERNEL GRID
  // do nothing kernel test
  dfloat nothingElapsed = nothingTest(stream, Ntests);
  nothingElapsed = nothingTest(stream, Ntests);
  
  // warm up call
  runBK5Kernel (stream, Nq, numElements, lambda,
		c_op,
		c_DofToDofD, c_oddDofToDofD, c_evenDofToDofD,
		c_solIn, c_solOut,
		mode);
  
#if USE_GRAPH==1
  // hip stream capture
  hipGraph_t graph;
  
  hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);

  for(int test=0;test<Ntests;++test){

    runMassBK5Kernel (stream, Nq, numElements, lambda,
		      c_op,
		      c_DofToDofD, c_oddDofToDofD, c_evenDofToDofD,
		      c_solIn, c_solOut);
  }
  
  hipStreamEndCapture(stream, &graph);
  
  hipGraphExec_t instance;
  hipGraphInstantiate(&instance, graph, NULL, NULL, 0);
#endif
  
  hipDeviceSynchronize();

  {
    hipEventRecord(start, stream);
    
#if USE_GRAPH==0
    for(int test=0;test<Ntests;++test){

      runBK5Kernel (stream, Nq, numElements, lambda,
		    c_op,
		    c_DofToDofD, c_oddDofToDofD, c_evenDofToDofD,
		    c_solIn, c_solOut,
		    mode);
      
    }
#else
    hipGraphLaunch(instance, stream);
#endif

    hipEventRecord(end, stream);

    hipDeviceSynchronize();
    
    float elapsed;
    hipEventElapsedTime(&elapsed, start, end);
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
  BK5Host (Nq, numElements, lambda, h_op, h_DofToDofD, h_solIn, h_solOut);

  // copy device version to host old q
  dfloat *fromDevice = (dfloat*) calloc(numElements*Np, sizeof(dfloat));
  hipMemcpy(fromDevice, c_solOut, numElements*Np*sizeof(dfloat), hipMemcpyDeviceToHost);

  dfloat maxDiff = 0;
  
  for(int e=0;e<numElements;++e){
    for(int n=0;n<Np;++n){
      int id = e*Np + n;
      dfloat diff = fabs(h_solOut[id]-fromDevice[id]);
      maxDiff = (diff>maxDiff) ? diff:maxDiff;
    }
  }
  printf("|| Mq_{host} - Mq_{device} ||_linf = %lg\n", maxDiff);
  
  hipEventDestroy(start);
  hipEventDestroy(end);	
  
  return 0;

}
