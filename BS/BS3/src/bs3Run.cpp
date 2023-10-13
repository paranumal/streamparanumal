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

#include "bs3.hpp"

void bs3_t::Run(){

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

  int sc = 1*sizeof(dfloat);  // bytes moved per entry
  int Nmin = Bmin/sc;
  int Nmax = Bmax/sc;
  int Nstep = (Bstep/sc > 0) ? Bstep/sc : 1;

  deviceMemory<dfloat> o_a    = platform.malloc<dfloat>(Nmax);
  deviceMemory<dfloat> o_tmp  = platform.malloc<dfloat>(blockSize);
  deviceMemory<dfloat> o_norm = platform.malloc<dfloat>(1);


  int Nwarm = 5;
  int Nblock = (Nmax+blockSize-1)/blockSize;
  Nblock = (Nblock>blockSize) ? blockSize : Nblock; //limit to blockSize entries
  int Nreads = (Nmax+Nblock*blockSize-1)/(Nblock*blockSize);
  for(int n=0;n<Nwarm;++n){ //warmup
    kernel1(Nblock, Nmax, Nreads, o_a, o_tmp); //partial reduction
    kernel2(Nblock, o_tmp, o_norm); //finish reduction

    if(n==0){
      memory<dfloat> tmp(1,0.);
      o_norm.copyTo(tmp);
      printf("CHECKSUM ERROR = %e\n", tmp[0]-Nmax);
    }
  }

  if (B) {
    //single test
    int N = B/sc;
    Nmin = N;
    Nmax = N;
    printf("BS3 = [");
  } else {
    //sweep test
    printf("%%Norm [DOFs, elapsed, DOFs/s, BW (GB/s)]\n");
    printf("BS3 = [\n");
  }

  //test
  for(int N=Nmin;N<=Nmax;N+=Nstep){
    timePoint_t start = GlobalPlatformTime(platform);

    /* NORM Test */
    int Ntests = 20;
    Nblock = (N+blockSize-1)/blockSize;
    Nblock = (Nblock>blockSize) ? blockSize : Nblock; //limit to blockSize entries
    Nreads = (N+Nblock*blockSize-1)/(Nblock*blockSize);
    for(int n=0;n<Ntests;++n){ //warmup
      kernel1(Nblock, N, Nreads, o_a, o_tmp); //partial reduction
      kernel2(Nblock, o_tmp, o_norm); //finish reduction
    }

    timePoint_t end = GlobalPlatformTime(platform);
    double elapsedTime = ElapsedTime(start, end)/Ntests;

    size_t bytesIn  = N*sizeof(dfloat);
    size_t bytesOut = 0;
    size_t bytes = bytesIn + bytesOut;

    printf("%d %5.4e %5.4e %6.2f",
            N, elapsedTime, N/elapsedTime, (double)(bytes/1.e9)/elapsedTime);
    if (N<Nmax) printf(";\n");
  }
  printf("]; %%Norm [DOFs, elapsed, DOFs/s, BW (GB/s)]\n");
}
