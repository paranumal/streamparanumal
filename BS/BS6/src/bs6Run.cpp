/*

The MIT License (MIT)

Copyright (c) 2020 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

#include "bs6.hpp"

void bs6_t::Run(){

  //create occa buffers
  dlong N = mesh.Np*mesh.Nelements;
  dlong Ngather = mesh.ogs.Ngather;
  deviceMemory<dfloat> o_q  = platform.malloc<dfloat>(N);
  deviceMemory<dfloat> o_gq = platform.malloc<dfloat>(Ngather);

  /* Warmup */
  for(int n=0;n<5;++n){
    mesh.ogs.Gather(o_gq, o_q, 1, ogs::Add, ogs::Trans); //dry run
  }

  /* Gather test */
  int Ntests = 20;
  timePoint_t start = GlobalPlatformTime(platform);

  for(int n=0;n<Ntests;++n){
    mesh.ogs.Gather(o_gq, o_q, 1, ogs::Add, ogs::Trans);
  }

  timePoint_t end = GlobalPlatformTime(platform);
  double elapsedTime = ElapsedTime(start, end)/Ntests;

  hlong NgatherGlobal = mesh.ogs.NgatherGlobal;
  hlong NGlobal = N;
  mesh.comm.Allreduce(NGlobal);

  size_t bytesIn=0;
  size_t bytesOut=0;
  bytesIn += (NgatherGlobal+1)*sizeof(dlong); //row starts
  bytesIn += NGlobal*sizeof(dlong); //local Ids
  bytesIn += NGlobal*sizeof(dfloat); //values
  bytesOut+= NgatherGlobal*sizeof(dfloat);

  size_t bytes = bytesIn + bytesOut;

  hlong Ndofs = mesh.ogs.NgatherGlobal;
  size_t Nflops = NGlobal;

  if ((mesh.rank==0)){
    printf("BS6 = [%d, " hlongFormat ", %5.4le, %5.4le, %6.2f, %6.2f]; %% Gather [N, DOFs, elapsed, DOFs/(ranks*s), avg BW (GB/s), avg GFLOPs] \n",
           mesh.N,
           Ndofs,
           elapsedTime,
           Ndofs/(mesh.size*elapsedTime),
           bytes/(1.0e9 * elapsedTime),
           Nflops/(1.0e9 * elapsedTime));
  }
}
