/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

#include "bs8.hpp"
#include "ogs.hpp"

typedef struct{
  int degree;
  dlong *ids;
}list_t;

void bs8_t::Run(){

  //create occa buffers
  dlong N = mesh.Np*mesh.Nelements;
  deviceMemory<dfloat> o_q = platform.malloc<dfloat>(N);

  mesh.ogs.ReorderGatherScatter();

#if 0
  deviceMemory<dfloat> o_gq = platform.malloc<dfloat>(mesh.ogs.Ngather);

  memory<int32_t> h_dest(N, -1);
  memory<int32_t> h_source(mesh.ogs.Ngather);
  
  for(int n=0;n<N;++n){
    h_dest[n] = -1;
  }
  for(int n=0;n<mesh.ogs.Ngather;++n){
    h_source[n] = n;
  }
  
  deviceMemory<int32_t> o_dest   = platform.malloc<int32_t>(h_dest);
  deviceMemory<int32_t> o_source = platform.malloc<int32_t>(h_source);
  
  mesh.ogs.Scatter(o_dest, o_source, 1, ogs::NoTrans);

  // OCCA build stuff
  properties_t kernelInfo = platform.props(); //copy base occa properties
  kernelInfo["defines/" "p_blockSize"] = (int)256;
  occa::kernel modScatterKernel = platform.buildKernel(DBS8 "/../BS7/okl/bs7.okl", "bs7", kernelInfo);
#endif  

  
  /* Warmup */
  for(int n=0;n<5;++n){
#if 1
    mesh.ogs.GatherScatter(o_q, 1, ogs::Add, ogs::Sym); //dry run
#else
    mesh.ogs.Gather(o_gq, o_q, 1, ogs::Add, ogs::Sym); //dry run
    //    mesh.ogs.Scatter(o_q, o_gq, 1, ogs::Sym); //dry run
    modScatterKernel(N, o_dest, o_gq, o_q);
#endif
  }

  /* Gather Scatter test */
  int Ntests = 50;
  timePoint_t start = GlobalPlatformTime(platform);

  for(int n=0;n<Ntests;++n){
#if 1
    mesh.ogs.GatherScatter(o_q, 1, ogs::Add, ogs::Sym);
#else
    mesh.ogs.Gather(o_gq, o_q, 1, ogs::Add, ogs::Sym); //dry run
    //    mesh.ogs.Scatter(o_q, o_gq, 1, ogs::Sym); //dry run
    modScatterKernel(N, o_dest, o_gq, o_q);
#endif
  }

  timePoint_t end = GlobalPlatformTime(platform);
  double elapsedTime = ElapsedTime(start, end)/Ntests;

  hlong NtotalGlobal = mesh.Nelements*mesh.Np;
  mesh.comm.Allreduce(NtotalGlobal);

  hlong NgatherGlobal = mesh.ogs.NgatherGlobal;

  size_t bytesIn=0;
  size_t bytesOut=0;
  //  bytesIn += (NgatherGlobal+1)*sizeof(dlong); //row starts
  bytesIn += (NgatherGlobal+1)*sizeof(dlong); //row starts
  bytesIn += NtotalGlobal*sizeof(dlong); //local Ids
  bytesIn += NtotalGlobal*sizeof(dfloat); //values
  bytesOut+= NtotalGlobal*sizeof(dfloat);

  size_t bytes = bytesIn + bytesOut;

  hlong NflopsGlobal = NtotalGlobal;

  hlong Ndofs = mesh.ogs.NgatherGlobal;

  if ((mesh.rank==0)){
    printf("BS8 = [%d, " hlongFormat ", %5.4le, %5.4le, %6.2f, %6.2f]; %% GatherScatter [N, DOFs, elapsed, DOFs/(ranks*s), avg BW (GB/s), avg GFLOPs] \n",
           mesh.N,
           Ndofs,
           elapsedTime,
           Ndofs/(mesh.size*elapsedTime),
           bytes/(1.0e9 * elapsedTime),
           NflopsGlobal/(1.0e9 * elapsedTime));
  }
}
