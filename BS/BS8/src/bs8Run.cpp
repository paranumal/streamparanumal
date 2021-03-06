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

  platform_t &platform = mesh.platform;

  //create occa buffers
  dlong N = mesh.Np*mesh.Nelements;
  occa::memory o_q = platform.malloc(N*sizeof(dfloat));

  /* Warmup */
  for(int n=0;n<5;++n){
    mesh.ogsMasked->GatherScatter(o_q, ogs_dfloat, ogs_add, ogs_sym); //dry run
  }

  /* Gather Scatter test */
  int Ntests = 50;
  platform.device.finish();
  MPI_Barrier(mesh.comm);
  double startTime = MPI_Wtime();

  for(int n=0;n<Ntests;++n){
    mesh.ogsMasked->GatherScatter(o_q, ogs_dfloat, ogs_add, ogs_sym);
  }

  platform.device.finish();
  MPI_Barrier(mesh.comm);
  double endTime = MPI_Wtime();
  double elapsedTime = (endTime - startTime)/Ntests;

  hlong Nblocks =    mesh.ogsMasked->symGatherScatter.NrowBlocks
                  +2*mesh.ogsMasked->haloScatter.NrowBlocks;
  hlong NblocksGlobal;
  MPI_Allreduce(&Nblocks, &NblocksGlobal, 1, MPI_HLONG, MPI_SUM, mesh.comm);

  hlong Ngather =    mesh.ogsMasked->symGatherScatter.Nrows
                  +2*mesh.ogsMasked->haloScatter.Nrows;
  hlong NgatherGlobal;
  MPI_Allreduce(&Ngather, &NgatherGlobal, 1, MPI_HLONG, MPI_SUM, mesh.comm);

  hlong NLocal  =    mesh.ogsMasked->symGatherScatter.nnz
                  +2*mesh.ogsMasked->haloScatter.nnz;
  hlong NGlobal;
  MPI_Allreduce(&NLocal, &NGlobal, 1, MPI_HLONG, MPI_SUM, mesh.comm);

  size_t bytesIn=0;
  size_t bytesOut=0;
  bytesIn += (NblocksGlobal+1)*sizeof(dlong); //block starts
  bytesIn += (NgatherGlobal+1)*sizeof(dlong); //row starts
  bytesIn += NGlobal*sizeof(dlong); //local Ids
  bytesIn += NGlobal*sizeof(dfloat); //values
  bytesOut+= NGlobal*sizeof(dfloat);

  size_t bytes = bytesIn + bytesOut;

  hlong Nflops =    mesh.ogsMasked->symGatherScatter.nnz
                  + mesh.ogsMasked->haloScatter.nnz;
  hlong NflopsGlobal;
  MPI_Allreduce(&Nflops, &NflopsGlobal, 1, MPI_HLONG, MPI_SUM, mesh.comm);

  hlong Ndofs = mesh.ogsMasked->NgatherGlobal;

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
