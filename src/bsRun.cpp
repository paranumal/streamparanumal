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

void bs_t::Run(){

  properties_t kernelInfo = platform.props(); //copy base occa properties
  tuningParams(kernelInfo);

  BuildKernel(kernelInfo);

  size_t B = 0;
  settings.getSetting("BYTES", B);
  AllocBuffers(B);

  std::string name = "BS" + std::to_string(problemNumber) + ", " + testName();

  if (problemNumber==0) {//KLL
    int Nlaunches=10;
    int Nwarmup=10;

    for(int n=0;n<Nwarmup;++n){
      RunKernel(N);
    }

    timePoint_t start = PlatformTime(platform);
    for(int n=0;n<Nlaunches;++n){
      RunKernel(N);
    }
    timePoint_t end = PlatformTime(platform);
    double elapsed = ElapsedTime(start, end);

    if (comm.rank()==0){
      printf("%s: Nlaunches=%d, time per launch=%5.4e, elapsed=%5.4e\n",
             name.c_str(), Nlaunches, elapsed/Nlaunches, elapsed);
    }
  } else if (problemNumber==1 ||
             problemNumber==2 ||
             problemNumber==3 ||
             problemNumber==4 ||
             problemNumber==5 ||
             problemNumber==6 ||
             problemNumber==7 ||
             problemNumber==8) {
    int Nwarmup=20;
    int Ntests=20;

    for(int n=0;n<Nwarmup;++n){
      RunKernel(N);
    }

    timePoint_t start = PlatformTime(platform);
    for(int n=0;n<Ntests;++n){
      RunKernel(N);
    }
    timePoint_t end = PlatformTime(platform);
    double elapsed = ElapsedTime(start, end)/Ntests;

    size_t bytes = bytesMoved(N);

    if (problemNumber==1 ||
        problemNumber==2 ||
        problemNumber==3 ||
        problemNumber==4 ||
        problemNumber==5) {
      if (comm.rank()==0){
        printf("%s: DOFs=%11ld, elapsed=%5.4e, BW (GB/s)=%6.2f, BW/rank (GB/s)=%6.2f\n",
               name.c_str(), static_cast<size_t>(Nglobal), elapsed,
               (double)(bytes/1.e9)/elapsed,
               (double)(bytes/1.e9)/(elapsed*comm.size()));
      }
    } else {
      if (comm.rank()==0){
        printf("%s: N=%d, DOFs=" hlongFormat ", %s, elapsed=%5.4e, BW (GB/s)=%6.2f, BW/rank (GB/s)=%6.2f\n",
               name.c_str(), mesh.N, NgatherGlobal,
               mesh.elementName().c_str(), elapsed,
               (double)(bytes/1.e9)/elapsed,
               (double)(bytes/1.e9)/(elapsed*comm.size()));
      }
    }
  }
}
