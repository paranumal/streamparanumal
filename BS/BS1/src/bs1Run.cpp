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

#include "bs1.hpp"

void bs1_t::Run(){

  //create arrays buffers
  size_t B = 0, Bmin = 0, Bmax = 0, Bstep = 0;
  settings.getSetting("BYTES", B);
  settings.getSetting("BMIN", Bmin);
  settings.getSetting("BMAX", Bmax);
  settings.getSetting("BSTEP", Bstep);

  //If nothing provide by user, default to single test with 1 GB of data
  if (!(B | Bmin | Bmax))
    B = 1073741824;

  if(B) Bmax = B;

  int sc = 2*sizeof(dfloat);  // bytes moved per entry
  int Nmin = Bmin/sc;
  int Nmax = Bmax/sc;
  int Nstep = (Bstep/sc > 0) ? Bstep/sc : 1;

  occa::memory o_a = platform.device.malloc(Nmax*sizeof(dfloat));
  occa::memory o_b = platform.device.malloc(Nmax*sizeof(dfloat));

  int Nwarm = 5;
  for(int n=0;n<Nwarm;++n){ //warmup
    kernel(Nmax, o_a, o_b); //b = a
  }

  if (B) {
    //single test
    int N = B/sc;
    Nmin = N;
    Nmax = N;
    printf("BS1 = [");
  } else {
    //sweep test
    printf("%%[DOFs, elapsed, DOFs/s, BW (GB/s)]\n");
    printf("BS1 = [\n");
  }

  //test
  for(int N=Nmin;N<=Nmax;N+=Nstep){
    // tic
    platform.device.finish();
    dfloat tic = MPI_Wtime();

    /* COPY Test */
    int Ntests = 20;
    for(int n=0;n<Ntests;++n){
      kernel(N, o_a, o_b); //b = a
    }

    platform.device.finish();
    dfloat toc = MPI_Wtime();
    double elapsedTime = (toc-tic)/Ntests;

    size_t bytesIn  = N*sizeof(dfloat);
    size_t bytesOut = N*sizeof(dfloat);
    size_t bytes = bytesIn + bytesOut;

    printf("%d %5.4e %5.4e %6.2f",
            N, elapsedTime, N/elapsedTime, (double)(bytes/1.e9)/elapsedTime);
    if (N<Nmax) printf(";\n");
  }
  printf("]; %%[DOFs, elapsed, DOFs/s, BW (GB/s)]\n");
}
