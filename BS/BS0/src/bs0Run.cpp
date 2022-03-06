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

#include "bs0.hpp"
#include "timer.hpp"

void bs0_t::Run(){

  int Nlaunches=10;
  int NlaunchMin=1;
  int NlaunchMax=1024;
  int NlaunchStep=1;
  settings.getSetting("NLAUNCHES", Nlaunches);
  settings.getSetting("NLAUNCHMIN", NlaunchMin);
  settings.getSetting("NLAUNCHMAX", NlaunchMax);
  settings.getSetting("NLAUNCHSTEP", NlaunchStep);

  //If nothing provide by user, default to single test of 10 launches
  if (!(Nlaunches | NlaunchMin | NlaunchMax))
    Nlaunches = 10;

  //create array buffers
  int N = 1;
  deviceMemory<dfloat> o_a = platform.malloc<dfloat>(N);
  deviceMemory<dfloat> o_b = platform.malloc<dfloat>(N);

  //warmup
  int Nwarm = 100;
  for(int w=0;w<Nwarm;++w){
    kernel(N, o_a, o_b);
  }

  if (Nlaunches) {
    //single test
    NlaunchMin = Nlaunches;
    NlaunchMax = Nlaunches;
    printf("BS0 = [");
  } else {
    //sweep test
    printf("%%Launch Latency [Nlaunches, time per launch, elapsed]\n");
    printf("BS0 = [\n");
  }

  //test
  for(Nlaunches=NlaunchMin;Nlaunches<=NlaunchMax;Nlaunches+=NlaunchStep){

    timePoint_t start = PlatformTime(platform);
    for(int n=0;n<Nlaunches;++n){
      kernel(N, o_a, o_b);
    }
    timePoint_t end = PlatformTime(platform);
    double elapsed = ElapsedTime(start, end);

    printf("%d %5.4e %5.4e", Nlaunches, elapsed/Nlaunches, elapsed);
    if (Nlaunches<NlaunchMax) printf(";\n");
  }

  printf("]; %%Launch Latency [Nlaunches, time per launch, elapsed]\n");
}
