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

#include "bs2.hpp"

void bs2_t::Run(){

  //create array buffers
  size_t B = 0, Bmin = 0, Bmax = 0, Bstep = 0;
  settings.getSetting("BYTES", B);
  settings.getSetting("BMIN", Bmin);
  settings.getSetting("BMAX", Bmax);
  settings.getSetting("BSTEP", Bstep);

  //If nothing provide by user, default to single test with 1 GB of data
  if (!(B | Bmin | Bmax))
    B = 1073741824;

  if(B) Bmax = B;

  int sc = 3*sizeof(dfloat);  // bytes moved per entry
  int Nmin = Bmin/sc;
  int Nmax = Bmax/sc;
  int Nstep = (Bstep/sc > 0) ? Bstep/sc : 1;

  deviceMemory<dfloat> o_a = platform.malloc<dfloat>(Nmax);
  deviceMemory<dfloat> o_b = platform.malloc<dfloat>(Nmax);

  const dfloat alpha = 1.0;
  const dfloat beta = 1.0;

  int Nwarm = 5;
  for(int n=0;n<Nwarm;++n){ //warmup
    kernel(Nmax, alpha, o_a, beta, o_b); //b = alpha*a + beta*b
  }

  if (B) {
    //single test
    int N = B/sc;
    Nmin = N;
    Nmax = N;
    printf("BS2 = [");
  } else {
    //sweep test
    printf("%%AXPY [DOFs, elapsed, DOFs/s, BW (GB/s)]\n");
    printf("BS2 = [\n");
  }

  //test
  for(int N=Nmin;N<=Nmax;N+=Nstep){
    timePoint_t start = GlobalPlatformTime(platform);

    /* AXPY Test */
    int Ntests = 20;
    for(int n=0;n<Ntests;++n){
      kernel(N, alpha, o_a, beta, o_b); //b = alpha*a + beta*b
    }

    timePoint_t end = GlobalPlatformTime(platform);
    double elapsedTime = ElapsedTime(start, end)/Ntests;

    size_t bytesIn  = 2*N*sizeof(dfloat);
    size_t bytesOut = N*sizeof(dfloat);
    size_t bytes = bytesIn + bytesOut;

    printf("%d %5.4e %5.4e %6.2f",
            N, elapsedTime, N/elapsedTime, (double)(bytes/1.e9)/elapsedTime);
    if (N<Nmax) printf(";\n");
  }
  printf("]; %%AXPY [DOFs, elapsed, DOFs/s, BW (GB/s)]\n");
}
