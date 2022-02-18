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
  dlong Ngather = mesh.ogs.Ngather;
  occa::memory o_q = platform.malloc(N*sizeof(dfloat));
  occa::memory o_gq = platform.malloc(Ngather*sizeof(dfloat));

  /* Warmup */
  for(int n=0;n<5;++n){
    mesh.ogs.Scatter(o_q, o_gq, 1, ogs::Dfloat, ogs::Add, ogs::NoTrans); //dry run
  }

  /* Scatter test */
  int Ntests = 20;
  platform.device.finish();
  MPI_Barrier(mesh.comm);
  double startTime = MPI_Wtime();

  for(int n=0;n<Ntests;++n){
    mesh.ogs.Scatter(o_q, o_gq, 1, ogs::Dfloat, ogs::Add, ogs::NoTrans);
  }

  platform.device.finish();
  MPI_Barrier(mesh.comm);
  double endTime = MPI_Wtime();
  double elapsedTime = (endTime - startTime)/Ntests;

  hlong Ntotal = mesh.Nelements*mesh.Np;
  hlong NtotalGlobal;
  MPI_Allreduce(&Ntotal, &NtotalGlobal, 1, MPI_HLONG, MPI_SUM, mesh.comm);

  hlong NgatherGlobal = mesh.ogs.NgatherGlobal;

  size_t bytesIn  = NgatherGlobal*sizeof(dfloat)+NtotalGlobal*sizeof(dlong);
  size_t bytesOut = NtotalGlobal*(sizeof(dfloat));
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
