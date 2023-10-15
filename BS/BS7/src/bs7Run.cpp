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

#include "bs7.hpp"
#include <unistd.h>
#include <iomanip>

void bs7_t::Run(){

  //create occa buffers
  dlong N = mesh.Np*mesh.Nelements;
  dlong Ngather = mesh.ogs.Ngather;

  deviceMemory<dfloat> o_q  = platform.malloc<dfloat>(N);
  deviceMemory<dfloat> o_gq = platform.malloc<dfloat>(Ngather);
  
  {
    
    memory<int32_t> h_dest(N, -1);
    memory<int32_t> h_source(Ngather);
    
    for(int n=0;n<N;++n){
      h_dest[n] = -1;
    }
    for(int n=0;n<Ngather;++n){
      h_source[n] = n;
    }
    
    deviceMemory<int32_t> o_dest   = platform.malloc<int32_t>(h_dest);
    deviceMemory<int32_t> o_source = platform.malloc<int32_t>(h_source);
    
    mesh.ogs.Scatter(o_dest, o_source, 1, ogs::NoTrans);

    // OCCA build stuff
    properties_t kernelInfo = platform.props(); //copy base occa properties
    kernelInfo["defines/" "p_blockSize"] = (int)256;
    occa::kernel kernel = platform.buildKernel(DBS7 "/okl/bs7.okl", "bs7", kernelInfo);

    kernel(N, o_dest, o_gq, o_q);

    usleep(1000);
    
    timePoint_t start = GlobalPlatformTime(platform);
    kernel(N, o_dest, o_gq, o_q);
    timePoint_t end = GlobalPlatformTime(platform);
    double elapsed = ElapsedTime(start, end);

    std::cout << std::
      setprecision(5); 
    
    std::cout << "N=" << N
	      << ", Ngather=" << Ngather
	      << ", Elapsed=" << elapsed
	      << ", Throughput=" << (Ngather*sizeof(dlong)+2*N*sizeof(dfloat))/(1.e9*elapsed)
	      << ", GB/s" << std::endl;

    std::cout << "NgatherGlobal=" << mesh.ogs.NgatherGlobal
	      << ", NtotalGlobal=" << mesh.Np*mesh.Nelements
	      << ", Ngather=" << Ngather
	      << ", N=" << N << std::endl;
  }
  
  /* Warmup */
  for(int n=0;n<5;++n){
    mesh.ogs.Scatter(o_q, o_gq, 1, ogs::NoTrans); //dry run
  }

  /* Scatter test */
  int Ntests = 20;
  timePoint_t start = GlobalPlatformTime(platform);

  for(int n=0;n<Ntests;++n){
    mesh.ogs.Scatter(o_q, o_gq, 1, ogs::NoTrans);
  }

  timePoint_t end = GlobalPlatformTime(platform);
  double elapsedTime = ElapsedTime(start, end)/Ntests;

  hlong NtotalGlobal = mesh.Nelements*mesh.Np;
  mesh.comm.Allreduce(NtotalGlobal);

  hlong NgatherGlobal = mesh.ogs.NgatherGlobal;

  size_t bytesIn  = 0;
  size_t bytesOut = 0;
  bytesIn += (NgatherGlobal+1)*sizeof(dlong); //row starts
  bytesIn += NtotalGlobal*sizeof(dlong); //local Ids
  bytesIn += NgatherGlobal*sizeof(dfloat); //values
  bytesOut+= NtotalGlobal*sizeof(dfloat);

  size_t bytes = bytesIn + bytesOut;

  hlong Ndofs = mesh.ogs.NgatherGlobal;
  size_t Nflops = 0;

  if ((mesh.rank==0)){
    printf("BS7 = [%d, " hlongFormat ", %5.4le, %5.4le, %6.2f, %6.2f]; %% Scatter [N, DOFs, elapsed, DOFs/(ranks*s), avg BW (GB/s), avg GFLOPs] \n",
           mesh.N,
           Ndofs,
           elapsedTime,
           Ndofs/(mesh.size*elapsedTime),
           bytes/(1.0e9 * elapsedTime),
           Nflops/(1.0e9 * elapsedTime));
  }
}
