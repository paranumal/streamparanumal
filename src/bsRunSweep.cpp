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

#include "bs.hpp"

void bs_t::RunSweep(){

  if (problemNumber==6 ||
      problemNumber==7 ||
      problemNumber==8) {
    LIBP_FORCE_ABORT("Sweep mode not suported for BS6, 7, or 8");
  }

  properties_t kernelInfo = platform.props(); //copy base occa properties
  tuningParams(kernelInfo);

  BuildKernel(kernelInfo);

  size_t B = 0;
  settings.getSetting("BYTES", B);
  AllocBuffers(B);

  if (problemNumber==0) {//KLL
    int NlaunchMax=1024;
    int Nwarmup=10;

    for(int n=0;n<Nwarmup;++n){
      RunKernel(N);
    }

    //sweep test
    if (comm.rank()==0) {
      printf("%% BS0, KLL [Nlaunches, time per launch, elapsed]\n");
      printf("BS0 = [\n");
    }

    //test
    for(int Nlaunches=0;Nlaunches<=NlaunchMax;Nlaunches++){

      timePoint_t start = PlatformTime(platform);
      for(int n=0;n<Nlaunches;++n){
        RunKernel(n);
      }
      timePoint_t end = PlatformTime(platform);
      double elapsed = ElapsedTime(start, end);

      if (comm.rank()==0){
        printf("%d %5.4e %5.4e", Nlaunches, elapsed/Nlaunches, elapsed);
        if (Nlaunches<NlaunchMax) printf(";\n");
      }
    }
    if (comm.rank()==0) {
      printf("]; %% BS0, KLL [Nlaunches, time per launch, elapsed]\n");
    }
  } else if (problemNumber==1 ||
             problemNumber==2 ||
             problemNumber==3 ||
             problemNumber==4 ||
             problemNumber==5) {

    int Nwarmup=20;
    int Ntests=20;

    size_t Bstep = 1048576;

    std::string name = "BS" + std::to_string(problemNumber) + ", " + testName();

    for(int n=0;n<Nwarmup;++n){
      RunKernel(N);
    }

    //sweep test
    if (comm.rank()==0) {
      printf("%% %s [DOFs, elapsed, DOFs/s, BW (GB/s), BW/rank (GB/s)]\n", name.c_str());
      printf("%s = [\n", name.c_str());
    }

    for(size_t b=Bstep;b<=B;b+=Bstep){
      int NB = bytesToArraySize(b);

      timePoint_t start = PlatformTime(platform);
      for(int n=0;n<Ntests;++n){
        RunKernel(NB);
      }
      timePoint_t end = PlatformTime(platform);
      double elapsed = ElapsedTime(start, end)/Ntests;

      size_t bytes = bytesMoved(NB);

      if (comm.rank()==0){
        printf("%d %5.4e %6.2f %6.2f",
               NB, elapsed,
               (double)(bytes/1.e9)/elapsed,
               (double)(bytes/1.e9)/(elapsed*comm.size()));
        if (b<B) printf(";\n");
      }
    }
    if (comm.rank()==0) {
      printf("]; %% %s [DOFs, elapsed, DOFs/s, BW (GB/s), BW/rank (GB/s)]\n", name.c_str());
    }
  }
}
