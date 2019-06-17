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

OCCA_CUDA_COMPILER_FLAGS='--use_fast_math' OCCA_VERBOSE=1  ./BK9  8 7 2000 CUDA 0 

running with OpenCL: [ 8^3 node velocity, 7^3 node pressure, 2000 elements, on device 0, platform 0 ]

OCCA_OPENCL_COMPILER_FLAGS='-cl-mad-enable -cl-finite-math-only -cl-fast-relaxed-math' OCCA_VERBOSE=1  ./BK9  8 7 2000 OpenCL 0 0

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

  // ./BK9 NqV NqP Nelements ThreadModel PlatformNumber DeviceNumber

  int NqV = atoi(argv[1]);
  int NqP = atoi(argv[2]);
  dlong Nelements = atoi(argv[3]);
  char *threadModel = strdup(argv[4]);

  int deviceId = 0;

  if(argc>=6)
    deviceId = atoi(argv[5]);
  
  int platformId = 0;
  if(argc>=7)
    platformId = atoi(argv[6]);

  printf("Running: NqV=%d, NqP=%d, Nelements=%d\n", NqV, NqP, Nelements);
  
  int NV = NqV-1;
  int NP = NqP-1;

  int NpV = NqV*NqV*NqV;
  int NpP = NqP*NqP*NqP;

  int Nvgeo = 10;
  int Ndim  = 3;
  
  dfloat lambda = 0;
  
  dlong offset = Nelements*NpV;

  // ------------------------------------------------------------------------------
  // build element nodes and operators
  
  dfloat *rV, *wV, *DrV;
  dfloat *rP, *wP, *DrP;
  dfloat *IPToV;

  meshJacobiGQ(0,0,NP, &rP, &wP);
  meshJacobiGQ(0,0,NV, &rV, &wV);
  
  meshDmatrix1D(NV, NqV, rV, &DrV);
  meshDmatrix1D(NP, NqP, rP, &DrP);

  meshInterpolationMatrix1D(NP, NqP, rP, NqV, rV, &IPToV);

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

  props["defines/p_NqV"] = NqV;
  props["defines/p_NqP"] = NqP;
  props["defines/p_NpV"] = NpV;
  props["defines/p_NpP"] = NpP;

  props["defines/p_Nvgeo"] = Nvgeo;
  props["defines/p_RXID"] = 0;
  props["defines/p_RYID"] = 1;
  props["defines/p_RZID"] = 2;
  props["defines/p_SXID"] = 3;
  props["defines/p_SYID"] = 4;
  props["defines/p_SZID"] = 5;
  props["defines/p_TXID"] = 6;
  props["defines/p_TYID"] = 7;
  props["defines/p_TZID"] = 8;
  props["defines/p_JWID"] = 9;

  props["defines/p_JWID"] = 9;

  props["defines/dfloat"] = dfloatString;
  props["defines/dlong"]  = dlongString;

  // ------------------------------------------------------------------------------
  // build kernel  
  occa::kernel BK9Kernel = device.buildKernel("BK9.okl", "BK9", props);

  // ------------------------------------------------------------------------------
  // populate device arrays

  dfloat *vgeo = drandAlloc(NpV*Nelements*Nvgeo);
  dfloat *eta  = drandAlloc(NpV*Nelements);
  dfloat *q    = drandAlloc((Ndim*NpV+NpP)*Nelements);
  dfloat *Aq   = drandAlloc((Ndim*NpV+NpP)*Nelements);

  occa::memory o_vgeo  = device.malloc(NpV*Nelements*Nvgeo*sizeof(dfloat), vgeo);
  occa::memory o_eta   = device.malloc(NpV*Nelements*sizeof(dfloat), eta);
  occa::memory o_q     = device.malloc((Ndim*NpV+NpP)*Nelements*sizeof(dfloat), q);
  occa::memory o_Aq    = device.malloc((Ndim*NpV+NpP)*Nelements*sizeof(dfloat), Aq);
  occa::memory o_DrV   = device.malloc(NqV*NqV*sizeof(dfloat), DrV);
  occa::memory o_DrP   = device.malloc(NqP*NqP*sizeof(dfloat), DrP);
  occa::memory o_IPToV = device.malloc(NqP*NqV*sizeof(dfloat), IPToV);

  occa::streamTag start, end;

  // warm up
  BK9Kernel(Nelements, offset, o_vgeo, o_DrV, o_IPToV, lambda, o_eta, o_q, o_Aq);

  device.finish();
  
  // run Ntests times
  int Ntests = 10;
  
  start = device.tagStream();

  for(int test=0;test<Ntests;++test)
    BK9Kernel(Nelements, offset, o_vgeo, o_DrV, o_IPToV, lambda, o_eta, o_q, o_Aq);
  
  end = device.tagStream();

  device.finish();

  double elapsed = device.timeBetween(start, end)/Ntests;

  long long int Ndofs = (NpV*Ndim+NpP)*Nelements;
  
  dfloat GnodesPerSecond = (NpV*Nelements/elapsed)/1.e9;
  dfloat GdofsPerSecond = (Ndofs/elapsed)/1.e9;
  
  printf("%02d %02d %06d %08d %08lld %e %e %e [NV, NP, Nelements, NVnodes, Ndofs, GVnodes/s, Gdofs/s, elapsed]\n",
	 NV,  NP, Nelements, NpV*Nelements, Ndofs, GnodesPerSecond, GdofsPerSecond, elapsed);
  
  return 0;
  
}
