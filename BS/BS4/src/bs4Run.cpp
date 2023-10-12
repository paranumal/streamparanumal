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

#include "bs4.hpp"

void bs4_t::Run(){

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

  int sc = 2*sizeof(dfloat);  // bytes moved per entry
  int Nmin = Bmin/sc;
  int Nmax = Bmax/sc;
  int Nstep = (Bstep/sc > 0) ? Bstep/sc : 1;

  libp::memory<dfloat> h_a(Nmax, 1.f);
  libp::memory<dfloat> h_b(Nmax, 1.f);
  
  deviceMemory<dfloat> o_a   = platform.malloc<dfloat>(Nmax, h_a);
  deviceMemory<dfloat> o_b   = platform.malloc<dfloat>(Nmax, h_b);
  deviceMemory<dfloat> o_tmp = platform.malloc<dfloat>(blockSize);
  deviceMemory<dfloat> o_dot = platform.malloc<dfloat>(1);

  int Nwarm = 5;
  int Nblock = (Nmax+blockSize-1)/blockSize;
  Nblock = (Nblock>blockSize) ? blockSize : Nblock; //limit to blockSize entries
  for(int n=0;n<Nwarm;++n){ //warmup
    int Nreads = (Nmax+(Nblock*blockSize)-1)/(Nblock*blockSize);
    kernel1(Nblock, Nmax, Nreads, o_a, o_b, o_tmp); //partial reduction
    kernel2(Nblock, o_tmp, o_dot); //finish reduction
    libp::memory<dfloat> h_tmp(1);
    o_dot.copyTo(h_tmp);
    printf("tmp=%g, Nmax=%d\n", h_tmp[0], Nmax);
  }

  if (B) {
    //single test
    int N = B/sc;
    Nmin = N;
    Nmax = N;
    printf("BS4 = [");
  } else {
    //sweep test
    printf("%%Dot [DOFs, elapsed, DOFs/s, BW (GB/s)]\n");
    printf("BS4 = [\n");
  }

  //test
  for(int N=Nmin;N<=Nmax;N+=Nstep){
    timePoint_t start = GlobalPlatformTime(platform);

    /* DOT Test */
    int Ntests = 20;
    Nblock = (N+blockSize-1)/blockSize;
    Nblock = (Nblock>blockSize) ? blockSize : Nblock; //limit to blockSize entries
    for(int n=0;n<Ntests;++n){ //warmup
      int Nreads = (N+Nblock*blockSize-1)/(Nblock*blockSize);
      kernel1(Nblock, N, Nreads, o_a, o_b, o_tmp); //partial reduction
      kernel2(Nblock, o_tmp, o_dot); //finish reduction
    }

    timePoint_t end = GlobalPlatformTime(platform);
    double elapsedTime = ElapsedTime(start, end)/Ntests;

    size_t bytesIn  = 2*N*sizeof(dfloat);
    size_t bytesOut = Nblock*sizeof(dfloat);
    size_t bytes = bytesIn + bytesOut;

    printf("%d %5.4e %5.4e %6.2f",
            N, elapsedTime, N/elapsedTime, (double)(bytes/1.e9)/elapsedTime);
    if (N<Nmax) printf(";\n");
  }
  printf("]; %%Dot [DOFs, elapsed, DOFs/s, BW (GB/s)]\n");
}
