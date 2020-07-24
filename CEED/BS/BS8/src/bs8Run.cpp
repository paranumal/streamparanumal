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

#include "bs8.hpp"

void bs8_t::Run(){

  //create occa buffers
  dlong N = mesh.Np*mesh.Nelements;
  dlong Ngather = mesh.ogs->Ngather;
  occa::memory o_q = mesh.device.malloc(N*sizeof(dfloat));
  occa::memory o_gq = mesh.device.malloc(Ngather*sizeof(dfloat));

  /* Scatter test */
  int Nwarm = 5;
  for(int n=0;n<Nwarm;++n){
    mesh.ogs->Scatter(o_q, o_gq, ogs_dfloat, ogs_add, ogs_notrans); //dry run
  }

  mesh.device.finish();
  usleep(1000);

  int Ntests = 10;

  occa::streamTag start = mesh.device.tagStream();

  for(int n=0;n<Ntests;++n){
    mesh.ogs->Scatter(o_q, o_gq, ogs_dfloat, ogs_add, ogs_notrans);
  }

  occa::streamTag end = mesh.device.tagStream();
  mesh.device.finish();

  double elapsedTime = mesh.device.timeBetween(start, end)/Ntests;

  size_t bytesIn=0;
  size_t bytesOut=0;

  dlong Nblocks = mesh.ogs->localScatter.NrowBlocks;
  bytesIn += (Nblocks+1)*sizeof(dlong); //block starts
  bytesIn += (Ngather+1)*sizeof(dlong); //row starts
  bytesIn += N*sizeof(dlong); //local Ids
  bytesIn += Ngather*sizeof(dfloat); //values
  bytesOut+= N*sizeof(dfloat);

  size_t bytes = bytesIn + bytesOut;

  printf("BS8: " dlongFormat ", %4.4f, %1.2e, %1.2e, %4.1f ; dofs, elapsed, time per DOF, DOFs/time, BW (GB/s) \n",
         mesh.Nelements*mesh.Np,
         elapsedTime,
         elapsedTime/(mesh.Np*mesh.Nelements),
         mesh.Nelements*((dfloat) mesh.Np/elapsedTime),
         bytes/(1e9*elapsedTime));

  o_q.free();
  o_gq.free();
}
