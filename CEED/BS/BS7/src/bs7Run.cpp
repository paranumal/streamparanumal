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

#include "bs7.hpp"

void bs7_t::Run(){

  //create occa buffers
  dlong N = mesh.Np*mesh.Nelements;
  dlong Ngather = mesh.ogsMasked->Ngather;
  occa::memory o_q = mesh.device.malloc(N*sizeof(dfloat));
  occa::memory o_gq = mesh.device.malloc(Ngather*sizeof(dfloat));

  /* Gather test */
  for(int n=0;n<5;++n){
    mesh.ogsMasked->Gather(o_gq, o_q, ogs_dfloat, ogs_add, ogs_trans); //dry run
  }

  int Ntests = 50;

  mesh.device.finish();
  MPI_Barrier(mesh.comm);
  double startTime = MPI_Wtime();

  for(int n=0;n<Ntests;++n){
    mesh.ogsMasked->Gather(o_gq, o_q, ogs_dfloat, ogs_add, ogs_trans);
  }

  mesh.device.finish();
  MPI_Barrier(mesh.comm);
  double endTime = MPI_Wtime();
  double elapsedTime = (endTime - startTime)/Ntests;

  hlong Nblocks = mesh.ogs->localScatter.NrowBlocks+mesh.ogs->haloScatter.NrowBlocks;
  hlong NblocksGlobal;
  MPI_Allreduce(&Nblocks, &NblocksGlobal, 1, MPI_HLONG, MPI_SUM, mesh.comm);

  hlong NgatherGlobal = mesh.ogsMasked->NgatherGlobal;

  hlong NunMasked = N - mesh.Nmasked;
  hlong NunMaskedGlobal;
  MPI_Allreduce(&NunMasked, &NunMaskedGlobal, 1, MPI_HLONG, MPI_SUM, mesh.comm);

  size_t bytesIn=0;
  size_t bytesOut=0;
  bytesIn += (NblocksGlobal+1)*sizeof(dlong); //block starts
  bytesIn += (NgatherGlobal+1)*sizeof(dlong); //row starts
  bytesIn += NunMaskedGlobal*sizeof(dlong); //local Ids
  bytesIn += NunMaskedGlobal*sizeof(dfloat); //values
  bytesOut+= NgatherGlobal*sizeof(dfloat);

  size_t bytes = bytesIn + bytesOut;

  hlong Ndofs = mesh.ogsMasked->NgatherGlobal;
  size_t Nflops = NunMaskedGlobal;

  if ((mesh.rank==0)){
    printf("BS7 (gather): %d, " hlongFormat ", %4.4f, %1.2e, %4.1f, %4.1f, %1.2e; N, DOFs, elapsed, time per DOF, avg BW (GB/s), avg GFLOPs, DOFs/ranks*time \n",
           mesh.N,
           Ndofs,
           elapsedTime,
           elapsedTime/(Ndofs),
           bytes/(1.0e9 * elapsedTime),
           Nflops/(1.0e9 * elapsedTime),
           Ndofs/(mesh.size*elapsedTime));
  }

  o_q.free();
  o_gq.free();
}
