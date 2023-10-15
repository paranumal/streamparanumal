#include <hip/hip_runtime.h>

#if p_knl==1
#define bs9_v1 bs9
#elif p_knl==2
#define bs9_v2 bs9
#elif p_knl==3
#define bs9_v3 bs9
#elif p_knl==4
#define bs9_v4 bs9
#endif


#if 0
// limited selection of sizes
using rfloat_t = __attribute__((__vector_size__(p_Nreads* sizeof(dfloat)))) dfloat;
using wfloat_t = __attribute__((__vector_size__(p_Nwrites* sizeof(dfloat)))) dfloat;

extern "C" __global__ __launch_bounds__(p_blockSize) void bs9_v0(const dlong N,
								 const rfloat_t* __restrict__ x,
								 wfloat_t*  __restrict__  y){
  
  const int n = threadIdx.x + blockIdx.x*blockDim.x;
  
  if(n<N){
    rfloat_t xn = x[n];
    //    rfloat_t xn = __builtin_nontemporal_load(x+n);
    dfloat res = 0;

#pragma unroll p_Nreads
    for(int m=0;m<p_Nreads;++m){
      res += xn[m];
    }

    wfloat_t yn;
#pragma unroll p_Nwrites
    for(int m=0;m<p_Nwrites;++m){
      yn[m] = res;
    }
    
    y[n] = yn;
    //
  }
}
#endif

// long stride
extern "C" __global__ __launch_bounds__(p_blockSize) void bs9_v1(const dlong N,
								 const dfloat* __restrict__ x,
								 dfloat*  __restrict__  y){
  
  const int n = threadIdx.x + blockIdx.x*blockDim.x;
  
  if(n<N){
    dfloat res = 0;
    
#pragma unroll p_Nreads
    for(int m=0;m<p_Nreads;++m){
      res += x[n + m*N];
    }
    
#pragma unroll p_Nwrites
    for(int m=0;m<p_Nwrites;++m){
      y[n+m*N] = res;
    }
  }
}

// blockSize stride
extern "C" __global__ __launch_bounds__(p_blockSize) void bs9_v2(const dlong N,
								 const dfloat* __restrict__ x,
								 dfloat*  __restrict__  y){
  
  int n = threadIdx.x + blockIdx.x*p_blockSize*p_Nreads;
  
  dfloat res = 0;
  
#pragma unroll p_Nreads
  for(int m=0;m<p_Nreads;++m){
    dlong id = n + m*p_blockSize;
    if(id<p_Nreads*N){
      res += x[id];
    }
  }
  
  n = threadIdx.x + blockIdx.x*p_blockSize*p_Nwrites;
  
#pragma unroll p_Nwrites
  for(int m=0;m<p_Nwrites;++m){
    dlong id = n + m*p_blockSize;
    if(id<p_Nwrites*N){
      y[id] = res;
    }
  }
}

// blockSize stride with nontemporal write
extern "C" __global__ __launch_bounds__(p_blockSize) void bs9_v3(const dlong N,
								 const dfloat* __restrict__ x,
								 dfloat*  __restrict__  y){
  
  int n = threadIdx.x + blockIdx.x*p_blockSize*p_Nreads;
  
  dfloat res = 0;
  
#pragma unroll p_Nreads
  for(int m=0;m<p_Nreads;++m){
    dlong id = n + m*p_blockSize;
    if(id<p_Nreads*N){
      res += x[id];
    }
  }
  
  n = threadIdx.x + blockIdx.x*p_blockSize*p_Nwrites;
  
#pragma unroll p_Nwrites
  for(int m=0;m<p_Nwrites;++m){
    dlong id = n + m*p_blockSize;
    if(id<p_Nwrites*N){
      //      y[id] = res;
      __builtin_nontemporal_store(res, y+id);
    }
  }
}


// uncoalesced
typedef struct{
  dfloat data[p_Nreads];
}read_t;

typedef struct{
  dfloat data[p_Nwrites];
}write_t;

extern "C" __global__ __launch_bounds__(p_blockSize) void bs9_v4(const dlong N,
								 const read_t* __restrict__ x,
								 write_t*  __restrict__  y){
  
  const int n = threadIdx.x + blockIdx.x*blockDim.x;
  
  if(n<N){
    read_t xn = x[n];
    //    rfloat_t xn = __builtin_nontemporal_load(x+n);
    dfloat res = 0;

#pragma unroll p_Nreads
    for(int m=0;m<p_Nreads;++m){
      res += xn.data[m];
    }

    write_t yn;
#pragma unroll p_Nwrites
    for(int m=0;m<p_Nwrites;++m){
      yn.data[m] = res;
    }
    
    y[n] = yn;
    //
  }
}
