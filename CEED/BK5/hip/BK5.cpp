/*

See LICENSE file.

*/

#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

#define dfloat_t double

void matrixPrint(int Nrows, int Ncols, dfloat_t *A, const char *mess){
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

#define p_Nggeo 7

#define p_G00ID 0
#define p_G01ID 1
#define p_G02ID 2
#define p_G11ID 3
#define p_G12ID 4
#define p_G22ID 5
#define p_GWJID 6


__constant__ dfloat_t const_DofToDofD[MAX_DOFS_1D*MAX_DOFS_1D];
__constant__ dfloat_t const_oddDofToDofD[MAX_HALF_DOFS_1D*MAX_HALF_DOFS_1D];
__constant__ dfloat_t const_evenDofToDofD[MAX_HALF_DOFS_1D*MAX_HALF_DOFS_1D];

void randAlloc(int N, dfloat_t **h_a, dfloat_t **c_a){

  *h_a = (dfloat_t*) calloc(N, sizeof(dfloat_t));

  for(int n=0;n<N;++n)
    h_a[0][n] = drand48();

  hipMalloc(c_a, N*sizeof(dfloat_t));

  hipMemcpy(c_a[0], h_a[0], N*sizeof(dfloat_t), hipMemcpyHostToDevice);

}

__global__ void nothingKernel(){  }


template <int NUM_DOFS_1D, int p_Nblock >
  __forceinline__ __device__ 
  void BK5Device(const int numElements,
		 const int element,
		 const dfloat_t lambda,
		 const dfloat_t * __restrict__ op,
		 const dfloat_t * __restrict__ DofToDofD,
		 const dfloat_t * __restrict__ oddDofToDofD,
		 const dfloat_t * __restrict__ evenDofToDofD,
		 dfloat_t s_p[p_Nblock][NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D],
		 dfloat_t * __restrict__ r_Ap){
  
  __shared__ dfloat_t s_Gpr[p_Nblock][NUM_DOFS_1D][NUM_DOFS_1D];
  __shared__ dfloat_t s_Gps[p_Nblock][NUM_DOFS_1D][NUM_DOFS_1D];
  
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

    dfloat_t G00 = 0, G01 =0, G02 =0, G11 =0, G12 =0, G22 =0, GWJ =0;
    
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
    
    dfloat_t pr = 0.f;
    dfloat_t ps = 0.f;
    dfloat_t pt = 0.f;

#pragma unroll
    for(int m = 0; m < NUM_DOFS_1D; m++) {
      int im = ijN(m,i,NUM_DOFS_1D);
      int jm = ijN(m,j,NUM_DOFS_1D);
      int km = ijN(m,k,NUM_DOFS_1D);
      pr += DofToDofD[im]*s_p[blk][k][j][m];
      ps += DofToDofD[jm]*s_p[blk][k][m][i];
      pt += const_DofToDofD[km]*s_p[blk][m][j][i];
    }

    __syncthreads();
    
    s_Gpr[blk][j][i] = (G00*pr + G01*ps + G02*pt);
    s_Gps[blk][j][i] = (G01*pr + G11*ps + G12*pt);
    
    dfloat_t Gpt = (G02*pr + G12*ps + G22*pt);
    
    dfloat_t Apk = GWJ*lambda*s_p[blk][k][j][i];
    
    __syncthreads();
    
#pragma unroll
    for(int m = 0; m < NUM_DOFS_1D; m++){
      int mi = ijN(i,m,NUM_DOFS_1D);
      int mj = ijN(j,m,NUM_DOFS_1D);
      int km = ijN(m,k,NUM_DOFS_1D);
      Apk     += DofToDofD[mi]*s_Gpr[blk][j][m];
      Apk     += DofToDofD[mj]*s_Gps[blk][m][i];
      r_Ap[m] += const_DofToDofD[km]*Gpt; // DT(m,k)*ut(i,j,k,e)
    }
    
    r_Ap[k] += Apk;
  }
  
}

template <int NUM_DOFS_1D, int p_Nblock >
__global__ void BK5ConstantKernel(const int numElements,
				  const dfloat_t lambda,
				  const dfloat_t * __restrict__ op,
				  const dfloat_t * __restrict__ DofToDofD,
				  const dfloat_t * __restrict__ oddDofToDofD,
				  const dfloat_t * __restrict__ evenDofToDofD,
				  const dfloat_t * __restrict__ solIn,
				  dfloat_t * __restrict__ solOut){
  
  __shared__ dfloat_t s_DofToDofD[NUM_DOFS_2D];
  __shared__ dfloat_t s_p[p_Nblock][NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D];
  
  dfloat_t r_Ap[NUM_DOFS_1D];

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
    (numElements, element, lambda, op, s_DofToDofD, const_oddDofToDofD, const_evenDofToDofD, s_p, r_Ap);
  
  if(element<numElements){
#pragma unroll
    for(int c=0;c<NUM_DOFS_1D;++c){
      int id = ijklN(a,b,c,element,NUM_DOFS_1D);
      solOut[id] = r_Ap[c];
    }
  }
}

void BK5Host(int NUM_DOFS_1D, int numElements, dfloat_t lambda,
	     const dfloat_t * __restrict__ op,
	     const dfloat_t * __restrict__ DofToDofD,
	     const dfloat_t * q,
	     dfloat_t *lapqout){

  
  dfloat_t Gqr[NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D];
  dfloat_t Gqs[NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D];
  dfloat_t Gqt[NUM_DOFS_1D][NUM_DOFS_1D][NUM_DOFS_1D];
  
  for(int element=0;element<numElements;++element){
    
    for(int k=0;k<NUM_DOFS_1D;++k){
      for(int j=0;j<NUM_DOFS_1D;++j){
	for(int i=0;i<NUM_DOFS_1D;++i){
	  
	  dfloat_t qr = 0;
	  dfloat_t qs = 0;
	  dfloat_t qt = 0;
	  
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
	  
	  const int gbase = element*p_Nggeo*NUM_DOFS_3D + ijkN(i,j,k,NUM_DOFS_1D);
	  
	  dfloat_t G00 = op[gbase+p_G00ID*NUM_DOFS_3D];
	  dfloat_t G01 = op[gbase+p_G01ID*NUM_DOFS_3D];
	  dfloat_t G02 = op[gbase+p_G02ID*NUM_DOFS_3D];
	  dfloat_t G11 = op[gbase+p_G11ID*NUM_DOFS_3D];
	  dfloat_t G12 = op[gbase+p_G12ID*NUM_DOFS_3D];
	  dfloat_t G22 = op[gbase+p_G22ID*NUM_DOFS_3D];
	  
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
	  
	  const int gbase = element*p_Nggeo*NUM_DOFS_3D + ijkN(i,j,k,NUM_DOFS_1D);
	  
	  dfloat_t GWJ = op[gbase+p_GWJID*NUM_DOFS_3D];
	  dfloat_t lapq = lambda*GWJ*q[kji];
	  
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
  
  dfloat_t *h_bwTest1, *c_bwTest1;
  dfloat_t *h_bwTest2, *c_bwTest2;
  
  randAlloc(bwNtotal/2, &h_bwTest1, &c_bwTest1);
  randAlloc(bwNtotal/2, &h_bwTest2, &c_bwTest2);
  
  hipDeviceSynchronize();
  hipEventRecord(start, stream);
  
  for(int test=0;test<Ntests/2;++test){
    hipMemcpy(c_bwTest2, c_bwTest1, (bwNtotal/2)*sizeof(dfloat_t), hipMemcpyDeviceToDevice);
    hipMemcpy(c_bwTest1, c_bwTest2, (bwNtotal/2)*sizeof(dfloat_t), hipMemcpyDeviceToDevice);
  }
  
  hipEventRecord(end, stream);
  hipEventSynchronize(end);
  hipDeviceSynchronize();

  float elapsed;
  hipEventElapsedTime(&elapsed, start, end);
  elapsed /= 1000.; // convert to s
  elapsed /= (double) Ntests;
  
  double estimatedActualDeviceBandwidth = (bwNtotal*sizeof(dfloat_t)/elapsed)/1.e9;
  
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
			  dfloat_t *h_OP,   dfloat_t **c_OP, dfloat_t **c_oddOP,  dfloat_t **c_evenOP){

  int HALF_COLS_OP = ((NUM_COLS_OP+1)/2);
  int HALF_ROWS_OP = ((NUM_ROWS_OP+1)/2);
  
  dfloat_t *X = (dfloat_t*) calloc(NUM_COLS_OP*NUM_COLS_OP, sizeof(dfloat_t));
  dfloat_t *invX = (dfloat_t*) calloc(NUM_COLS_OP*NUM_COLS_OP, sizeof(dfloat_t));

  dfloat_t *cubX = (dfloat_t*) calloc(NUM_ROWS_OP*NUM_ROWS_OP, sizeof(dfloat_t));
  dfloat_t *cubInvX = (dfloat_t*) calloc(NUM_ROWS_OP*NUM_ROWS_OP, sizeof(dfloat_t));

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
  
  dfloat_t *IinvX = (dfloat_t*) calloc(NUM_COLS_OP*NUM_ROWS_OP, sizeof(dfloat_t));
  dfloat_t *cubInvXIinvX = (dfloat_t*) calloc(NUM_COLS_OP*NUM_ROWS_OP, sizeof(dfloat_t));

  // post multiply by invX
  for(int i=0;i<NUM_ROWS_OP;++i){
    for(int a=0;a<NUM_COLS_OP;++a){
      dfloat_t resI = 0;
      for(int n=0;n<NUM_COLS_OP;++n){
	resI += h_OP [i*NUM_COLS_OP+n]*invX[n*NUM_COLS_OP+a];
      }
      IinvX[i*NUM_COLS_OP+a] = resI;
    }
  }
  
  // pre multiply by invX
  for(int i=0;i<NUM_ROWS_OP;++i){
    for(int a=0;a<NUM_COLS_OP;++a){
      dfloat_t resI = 0;
      for(int n=0;n<NUM_ROWS_OP;++n){
	resI += cubInvX[i*NUM_ROWS_OP+n]*IinvX[n*NUM_COLS_OP + a];
      }
      cubInvXIinvX[i*NUM_COLS_OP+a] = resI;
    }
  }
  
  // now interleave the two non-zero blocks
  // [ A 0 ]  => [ A[0][0] B[0][0] A[0][1] B[0][1] .. A[0][HALF_DOFS_1D-1] B[0][HALF_DOFS_1D-1] .. 
  // [ 0 B ] 

  dfloat_t *oddOP  = (dfloat_t*) calloc(NUM_ROWS_OP*HALF_ROWS_OP, sizeof(dfloat_t));
  dfloat_t *evenOP = (dfloat_t*) calloc(NUM_ROWS_OP*HALF_ROWS_OP, sizeof(dfloat_t));
  
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
  
  hipMalloc(c_oddOP, NoddOP*sizeof(dfloat_t));
  hipMalloc(c_evenOP, NevenOP*sizeof(dfloat_t));
  
  hipMemcpy(*c_oddOP,  oddOP,  NoddOP*sizeof(dfloat_t),  hipMemcpyHostToDevice);
  hipMemcpy(*c_evenOP, evenOP, NoddOP*sizeof(dfloat_t), hipMemcpyHostToDevice);

  hipMemcpy(*c_OP, h_OP,  NUM_COLS_OP*NUM_ROWS_OP*sizeof(dfloat_t),  hipMemcpyHostToDevice);

  matrixPrint(NUM_COLS_OP, NUM_COLS_OP, X, "X");
  matrixPrint(NUM_ROWS_OP, NUM_ROWS_OP, cubX, "cubX");

  
  matrixPrint(NUM_COLS_OP, NUM_COLS_OP, invX, "invX");
  matrixPrint(NUM_ROWS_OP, NUM_ROWS_OP, cubInvX, "cubInvX");


  
}


void runBK5Kernel(hipStream_t stream, int Nq, int numElements, dfloat_t lambda,
		  dfloat_t *c_op,
		  dfloat_t *c_DofToDofD, dfloat_t *c_oddDofToDofD, dfloat_t *c_evenDofToDofD,
		  dfloat_t *c_solIn, dfloat_t *c_solOut){
  
#define BK5Kernel(Nq,Nblock)						\
  {									\
    dim3 G((numElements+Nblock-1)/Nblock, 1, 1);			\
    dim3 B(Nq*Nq, Nblock, 1);						\
    hipLaunchKernelGGL(BK5ConstantKernel<Nq,Nblock>, G, B, 0, stream,	\
		       numElements, lambda, c_op, c_DofToDofD, c_oddDofToDofD,c_evenDofToDofD, c_solIn, c_solOut); \
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


dfloat_t nothingTest(hipStream_t stream, int Ntests){

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
  
  if(argc!=3){
    printf("Usage: ./massMatrixMultiplyVT Nq numElements\n");
    exit(-1);
  }

  // read number of elements
  int          Nq = atoi(argv[1]);
  int numElements = atoi(argv[2]);

  dfloat_t lambda = 0;
  
  printf("Running: NUM_DOFS_1D=%d, numElements=%d\n", Nq, numElements);

  int   Np = Nq*Nq*Nq;
  int halfNq = ((Nq+1)/2);

  int    Ntotal = numElements*Np;

  int Ntests = 100;

  
  double estimatedActualDeviceBandwidth = bandwidthTest(stream, Ntests, (Ntotal*2+7*Ntotal)*sizeof(dfloat_t));
  
  dfloat_t *h_op,      *c_op;
  dfloat_t *h_solOut,       *c_solOut;
  dfloat_t *h_solIn,        *c_solIn;

  dfloat_t *h_DofToDofD,    *c_DofToDofD;
  dfloat_t *c_oddDofToDofD, *c_evenDofToDofD;

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

  hipMemcpyToSymbol(HIP_SYMBOL(const_DofToDofD),     c_DofToDofD,     Nq*Nq*sizeof(dfloat_t), 0, hipMemcpyDeviceToDevice);
  hipMemcpyToSymbol(HIP_SYMBOL(const_oddDofToDofD),  c_oddDofToDofD,  halfNq*halfNq*sizeof(dfloat_t), 0, hipMemcpyDeviceToDevice);
  hipMemcpyToSymbol(HIP_SYMBOL(const_evenDofToDofD), c_evenDofToDofD, halfNq*halfNq*sizeof(dfloat_t), 0, hipMemcpyDeviceToDevice);
  
  hipEvent_t start, end;
  hipEventCreate(&start);
  hipEventCreate(&end);	

  // KERNEL GRID
  // do nothing kernel test
  dfloat_t nothingElapsed = nothingTest(stream, Ntests);
  nothingElapsed = nothingTest(stream, Ntests);
  
  // warm up call
  runBK5Kernel (stream, Nq, numElements, lambda,
		c_op,
		c_DofToDofD, c_oddDofToDofD, c_evenDofToDofD,
		c_solIn, c_solOut);
  
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
		    c_solIn, c_solOut);
      
    }
#else
    hipGraphLaunch(instance, stream);
#endif

    hipEventRecord(end, stream);
    
    hipEventSynchronize(end);
    
    float elapsed;
    hipEventElapsedTime(&elapsed, start, end);
    elapsed /= 1000.;
    elapsed /= (double) Ntests;

    int bytesMoved = (2*Np+7*Np)*sizeof(dfloat_t); // x, Mx, opa   
    double bw = (bytesMoved*numElements/elapsed)/1.e9;
    
    printf("%2d %8d %8d %e %e %e %e %e %%%% [BK5: N, numElements, Ndofs,"
	   " elapsed, dofsPerSecond, nothingElapsed, BW in GB/s, estimatedActualDeviceBandwidth]\n",
	   Nq-1, numElements, Np*numElements, elapsed, numElements*(Np/elapsed),
	   nothingElapsed, bw, estimatedActualDeviceBandwidth);
  }

  // check output is correct
  BK5Host (Nq, numElements, lambda, h_op, h_DofToDofD, h_solIn, h_solOut);

  // copy device version to host old q
  dfloat_t *fromDevice = (dfloat_t*) calloc(numElements*Np, sizeof(dfloat_t));
  hipMemcpy(fromDevice, c_solOut, numElements*Np*sizeof(dfloat_t), hipMemcpyDeviceToHost);

  dfloat_t maxDiff = 0;
  
  for(int e=0;e<numElements;++e){
    for(int n=0;n<Np;++n){
      int id = e*Np + n;
      dfloat_t diff = fabs(h_solOut[id]-fromDevice[id]);
      maxDiff = (diff>maxDiff) ? diff:maxDiff;
    }
  }
  printf("|| Mq_{host} - Mq_{device} ||_linf = %lg\n", maxDiff);
  
  hipEventDestroy(start);
  hipEventDestroy(end);	
  
  return 0;

}
