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

OCCA_CUDA_COMPILER_FLAGS='--use_fast_math' OCCA_VERBOSE=1  ./BK1  8 7 2000 CUDA 0 

running with OpenCL: [ 8^3 node velocity, 7^3 node pressure, 2000 elements, on device 0, platform 0 ]

OCCA_OPENCL_COMPILER_FLAGS='-cl-mad-enable -cl-finite-math-only -cl-fast-relaxed-math' OCCA_VERBOSE=1  ./BK1  8 7 2000 OpenCL 0 0

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

  // ./BK1 Nq cubNq Nelements ThreadModel PlatformNumber DeviceNumber

  if(argc<5){
    printf("Usage: ./BK1 Nq cubNq Nelements threadModel deviceId platformId\n");
    exit(-1);
  }
  
  int Nq = atoi(argv[1]);
  int cubNq = atoi(argv[2]);
  dlong Nelements = atoi(argv[3]);
  char *threadModel = strdup(argv[4]);

  int deviceId = 0;

  if(argc>=6)
    deviceId = atoi(argv[5]);
  
  int platformId = 0;
  if(argc>=7)
    platformId = atoi(argv[6]);

  printf("Running: Nq=%d, cubNq=%d, Nelements=%d\n", Nq, cubNq, Nelements);
  
  int N = Nq-1;
  int Np= Nq*Nq*Nq;

  int cubN = cubNq-1;
  int cubNp = cubNq*cubNq*cubNq;

  int Nggeo = 1;
  
  // ------------------------------------------------------------------------------
  // build element nodes and operators
  
  dfloat *r, *w;
  dfloat *cubr, *cubw;
  dfloat *INToC;
  
  meshJacobiGL(0,0,N, &r, &w);
  meshJacobiGQ(0,0,cubN, &cubr, &cubw);
  
  meshInterpolationMatrix1D(N, Nq, r, cubNq, cubr, &INToC);

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
  props["defines/p_cubNq"] = cubNq;
  props["defines/p_Np"] = Np;
  props["defines/p_cubNp"] = cubNp;

  props["defines/p_Nggeo"] = Nggeo;
  props["defines/p_GWJID"] = 0;
  
  props["defines/dfloat"] = dfloatString;
  props["defines/dlong"]  = dlongString;

  // ------------------------------------------------------------------------------
  // build kernel  
  occa::kernel BK1Kernel = device.buildKernel("BK1.okl", "BK1", props);

  // ------------------------------------------------------------------------------
  // populate device arrays

  dfloat *ggeo = drandAlloc(cubNp*Nelements*Nggeo);
  dfloat *q    = drandAlloc(Np*Nelements);
  dfloat *Aq   = drandAlloc(Np*Nelements);

  occa::memory o_ggeo  = device.malloc(cubNp*Nelements*Nggeo*sizeof(dfloat), ggeo);
  occa::memory o_q     = device.malloc(Np*Nelements*sizeof(dfloat), q);
  occa::memory o_Aq    = device.malloc(Np*Nelements*sizeof(dfloat), Aq);
  occa::memory o_INToC = device.malloc(Nq*cubNq*sizeof(dfloat), INToC);

  occa::streamTag start, end;

  // --------------------------------------------------------------------------------
  // Test code against reference cpu (also acts as warm up)
  BK1Kernel(Nelements, o_ggeo, o_INToC, o_q, o_Aq);

  // compute reference solution
  meshReferenceBK1(Nq, cubNq, Nelements, ggeo, INToC, q, Aq);

  // compare occa and cpu solutions
  o_Aq.copyTo(q);
  dfloat maxDiff = 0;
  for(int n=0;n<Np*Nelements;++n){
    dfloat diff = fabs(q[n]-Aq[n]);
    maxDiff = (maxDiff<diff) ? diff:maxDiff;
  }
  printf(" |cpu(Aq)-gpu(Aq)|_linf = % e\n", maxDiff);
  
  device.finish();

  // --------------------------------------------------------------------------------
  // Run multiple tests
  
  // run Ntests times
  int Ntests = 10;

  start = device.tagStream();

  for(int test=0;test<Ntests;++test)
    BK1Kernel(Nelements, o_ggeo, o_INToC, o_q, o_Aq);

  end = device.tagStream();

  device.finish();

  double elapsed = device.timeBetween(start, end)/Ntests;

  long long int Ndofs = Np*Nelements;

  dfloat GnodesPerSecond = (Np*Nelements/elapsed)/1.e9;

  printf("%02d %02d %06d %08d %e %e "
	 "[N, cubN, Nelements, Nnodes, Gnodes/s, elapsed]\n",
	 N,  cubN, Nelements, Np*Nelements, GnodesPerSecond, elapsed);

  return 0;

}
