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
  occa::memory o_q = mesh.device.malloc(N*sizeof(dfloat));

  /* Gather Scatter test */
  for(int n=0;n<5;++n){
    mesh.ogsMasked->GatherScatter(o_q, ogs_dfloat, ogs_add, ogs_sym); //dry run
  }

  int Ntests = 50;

  occa::streamTag start = mesh.device.tagStream();

  for(int n=0;n<Ntests;++n){
    mesh.ogsMasked->GatherScatter(o_q, ogs_dfloat, ogs_add, ogs_sym);
  }

  occa::streamTag end = mesh.device.tagStream();
  mesh.device.finish();

  double elapsedTime = mesh.device.timeBetween(start, end)/Ntests;

  size_t bytesIn=0;
  size_t bytesOut=0;

  dlong Nblocks = mesh.ogsMasked->symGatherScatter.NrowBlocks;
  dlong Ngather = mesh.ogsMasked->symGatherScatter.Nrows;
  dlong Nlocal  = mesh.ogsMasked->symGatherScatter.nnz;

  bytesIn += (Nblocks+1)*sizeof(dlong); //block starts
  bytesIn += (Ngather+1)*sizeof(dlong); //row starts
  bytesIn += Nlocal*sizeof(dlong); //local Ids
  bytesIn += Nlocal*sizeof(dfloat); //values
  bytesOut+= Nlocal*sizeof(dfloat);

  size_t bytes = bytesIn + bytesOut;

  printf("BS6: " dlongFormat ", %4.4f, %1.2e, %1.2e, %4.1f ; dofs, elapsed, time per DOF, DOFs/time, BW (GB/s) \n",
         mesh.Nelements*mesh.Np,
         elapsedTime,
         elapsedTime/(mesh.Np*mesh.Nelements),
         mesh.Nelements*((dfloat) mesh.Np/elapsedTime),
         bytes/(1e9*elapsedTime));

  o_q.free();
}
