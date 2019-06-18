/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/


/* 

compilation: make

running with CUDA: [ 8^3 node velocity, 7^3 node pressure, 2000 elements, on device 0 ]

OCCA_CUDA_COMPILER_FLAGS='--use_fast_math' OCCA_VERBOSE=1  ./BK5  8 7 2000 CUDA 0 

running with OpenCL: [ 8^3 node velocity, 7^3 node pressure, 2000 elements, on device 0, platform 0 ]

OCCA_OPENCL_COMPILER_FLAGS='-cl-mad-enable -cl-finite-math-only -cl-fast-relaxed-math' OCCA_VERBOSE=1  ./BK5  8 7 2000 OpenCL 0 0

*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"
#include <unistd.h>
#include  "mpi.h"
#include "occa.hpp"
#include "meshBasis.hpp"

dfloat *drandAlloc(int N){

  dfloat *v = (dfloat*) calloc(N, sizeof(dfloat));

  for(int n=0;n<N;++n){
    v[n] = drand48();
  }

  return v;
}

  

int main(int argc, char **argv){

  // ./BK5 Nq Nelements ThreadModel PlatformNumber DeviceNumber

  if(argc<3){
    printf("Usage: ./BK5 Nq Nelements threadModel deviceId platformId\n");
    exit(-1);
  }
  
  int Nq = atoi(argv[1]);
  dlong Nelements = atoi(argv[2]);
  char *threadModel = strdup(argv[3]);

  int deviceId = 0;

  if(argc>=5)
    deviceId = atoi(argv[4]);
  
  int platformId = 0;
  if(argc>=6)
    platformId = atoi(argv[5]);

  printf("Running: Nq=%d, Nelements=%d\n", Nq, Nelements);
  
  int N = Nq-1;
  int Np = Nq*Nq*Nq;
  int Nggeo = 7;
  int Ndim  = 3;
  
  dfloat lambda = 0;
  
  dlong offset = Nelements*Np;

  // ------------------------------------------------------------------------------
  // build element nodes and operators
  
  dfloat *rV, *wV, *DrV;

  meshJacobiGQ(0,0,N, &rV, &wV);
  meshDmatrix1D(N, Nq, rV, &DrV);

  // ------------------------------------------------------------------------------
  // build device
  occa::device device;

  char deviceConfig[BUFSIZ];

  if(strstr(threadModel, "CUDA")){
    sprintf(deviceConfig, "mode: 'CUDA', device_id: %d",deviceId);
  }
  else if(strstr(threadModel,  "HIP")){
    sprintf(deviceConfig, "mode: 'HIP', device_id: %d",deviceId);
  }
  else if(strstr(threadModel,  "OpenCL")){
    sprintf(deviceConfig, "mode: 'OpenCL', device_id: %d, platform_id: %d", deviceId, platformId);
  }
  else if(strstr(threadModel,  "OpenMP")){
    sprintf(deviceConfig, "mode: 'OpenMP' ");
  }
  else{
    sprintf(deviceConfig, "mode: 'Serial' ");
  }

  std::string deviceConfigString(deviceConfig);
  
  device.setup(deviceConfigString);

  // ------------------------------------------------------------------------------
  // build kernel defines
  
  occa::properties props;
  props["defines"].asObject();
  props["includes"].asArray();
  props["header"].asArray();
  props["flags"].asObject();

  props["defines/p_Nq"] = Nq;
  props["defines/p_Np"] = Np;

  props["defines/p_Nggeo"] = Nggeo;
  props["defines/p_G00ID"] = 0;
  props["defines/p_G01ID"] = 1;
  props["defines/p_G02ID"] = 2;
  props["defines/p_G11ID"] = 3;
  props["defines/p_G12ID"] = 4;
  props["defines/p_G22ID"] = 5;
  props["defines/p_GWJID"] = 6;

  props["defines/dfloat"] = dfloatString;
  props["defines/dlong"]  = dlongString;

  // ------------------------------------------------------------------------------
  // build kernel  
  occa::kernel BK5Kernel = device.buildKernel("BK5.okl", "BK5", props);

  // ------------------------------------------------------------------------------
  // populate device arrays

  dfloat *ggeo = drandAlloc(Np*Nelements*Nggeo);
  dfloat *q    = drandAlloc((Ndim*Np)*Nelements);
  dfloat *Aq   = drandAlloc((Ndim*Np)*Nelements);

  dlong *elementList = (dlong*) calloc(Nelements, sizeof(dlong));
  for(dlong e=0;e<Nelements;++e)
    elementList[e] = e;
  
  occa::memory o_ggeo  = device.malloc(Np*Nelements*Nggeo*sizeof(dfloat), ggeo);
  occa::memory o_q     = device.malloc((Ndim*Np)*Nelements*sizeof(dfloat), q);
  occa::memory o_Aq    = device.malloc((Ndim*Np)*Nelements*sizeof(dfloat), Aq);
  occa::memory o_DrV   = device.malloc(Nq*Nq*sizeof(dfloat), DrV);
  occa::memory o_elementList  = device.malloc(Nelements*sizeof(dfloat), elementList);

  occa::streamTag start, end;

  // warm up
  BK5Kernel(Nelements, o_elementList, o_ggeo, o_DrV, lambda, o_q, o_Aq);

  device.finish();
  
  // run Ntests times
  int Ntests = 10;
  
  start = device.tagStream();

  for(int test=0;test<Ntests;++test)
    BK5Kernel(Nelements, o_elementList, o_ggeo, o_DrV, lambda, o_q, o_Aq);
  
  end = device.tagStream();

  device.finish();

  double elapsed = device.timeBetween(start, end)/Ntests;
  
  dfloat GnodesPerSecond = (Np*Nelements/elapsed)/1.e9;
  
  printf("%02d %06d %08d %e %e [N, Nelements, Nnodes, Gnodes/s, elapsed]\n",
	 N,  Nelements, Np*Nelements, GnodesPerSecond, elapsed);
  
  return 0;
  
}
