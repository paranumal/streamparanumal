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
  int N = 0;
  int Nmin = 0, Nmax = 0, Nsamples = 1;
  int B = 0, Bmin = 0, Bmax = 0, Bstep = 0;

  settings.getSetting("BYTES", B);
  if(B){
    Bmin = B;
    Bmax = B;
    Nsamples = 1;
  }
  else{
    settings.getSetting("BMIN", Bmin);
    settings.getSetting("BMAX", Bmax);
    settings.getSetting("NSAMPLES", Nsamples);
  }

  int sc = 2*sizeof(dfloat);  // bytes moved per entry
  Nmin = Bmin/sc;
  Nmax = Bmax/sc;
  N = Nmax;

  occa::memory o_a = platform.device.malloc(N*sizeof(dfloat));
  occa::memory o_b = platform.device.malloc(N*sizeof(dfloat));

  int Nwarm = 5;
  for(int n=0;n<Nwarm;++n){ //warmup
    kernel(N, o_a, o_b); //b = a
  }

  //  for(int test=0;test<1000000;++test){
  //    int Nrun = Nmax;
  printf("%%%% BS id, dofs, elapsed, time per DOF, DOFs/time, BW (GB/s) \n");
  for(int samp=1;samp<=Nsamples;++samp){
    int Nrun = mymin(Nmax, Nmin + (Nmax-Nmin)*((samp+1)*(samp+2)/(double)((Nsamples+1)*(Nsamples+2))));

    double minElapsedTime = 1e9;
    int Nattempts = 5;

    for(int att=0;att<Nattempts;++att){

      // tic
      platform.device.finish();
      dfloat tic = MPI_Wtime();

      /* COPY Test */
      int Ntests = 20;
      for(int n=0;n<Ntests;++n){
	kernel(Nrun, o_a, o_b); //b = a
      }

      platform.device.finish();
      dfloat toc = MPI_Wtime();
      double elapsedTime = (toc-tic)/Ntests;

      minElapsedTime = mymin(minElapsedTime, elapsedTime);
    }

    size_t bytesIn  = Nrun*sizeof(dfloat);
    size_t bytesOut = Nrun*sizeof(dfloat);
    size_t bytes = bytesIn + bytesOut;

    //    printf("1, " dlongFormat ", %1.5le, %1.5le, %1.5le, %1.5lef ;\n",
    //	   Nrun, minElapsedTime, minElapsedTime/Nrun, ((dfloat) Nrun)/minElapsedTime, bytes/(1e9*minElapsedTime));

    double Tlist[3], freqList[3];

    printf("1, " dlongFormat ", %1.5le, %1.5le, %1.5le, %1.5le, %1.5le, %1.5le, %1.5le, %1.5le ;\n",
	   Nrun, (double)minElapsedTime, (double)minElapsedTime/Nrun, ((dfloat) Nrun)/minElapsedTime, (double)(bytes/1.e9)/minElapsedTime,
	   Tlist[0], Tlist[1], Tlist[2], freqList[0]);

    //    fflush(stdout);
  }

  o_a.free();
  o_b.free();
}
