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

  //create arrays buffers
#if 0
  int N = 0;
  settings.getSetting("BYTES", N);
  N /= sizeof(dfloat);
#else
  int N = 0;
  int Nmin = 0, Nmax = 0, Nstep = 0;
  int B = 0, Bmin = 0, Bmax = 0, Bstep = 0;
  settings.getSetting("BYTES", B);
  if(B){
    Bmin = B;
    Bmax = B;
    Bstep = sizeof(dfloat);
  }
  else{
    settings.getSetting("BMIN", Bmin);
    settings.getSetting("BMAX", Bmax);
    settings.getSetting("BSTEP", Bstep);
  }
  // should scale down by #reads + #writes per entry
  N = Bmax/sizeof(dfloat);
  Nmax = Bmax/sizeof(dfloat);
  Nmin = Bmin/sizeof(dfloat);
  Nstep = Bstep/sizeof(dfloat);
#endif
  
  occa::memory o_a = device.malloc(N*sizeof(dfloat));
  occa::memory o_b = device.malloc(N*sizeof(dfloat));
  occa::memory o_c = device.malloc(N*sizeof(dfloat));

  for(int Nrun=Nmin;Nrun<=Nmax;Nrun+=Nstep){

    int Ntests = 10;
    int Nwarm = 5;
    for(int n=0;n<Nwarm;++n){ //warmup
      kernel(N, o_a, o_b, o_c); //c = a + b
    }
    
    // let GPU rest
    device.finish();

    /* ADD Test */
    occa::streamTag start = device.tagStream();
    
    for(int n=0;n<Ntests;++n){
      kernel(Nrun, o_a, o_b, o_c); //c = a + b
    }
    
    occa::streamTag end = device.tagStream();
    device.finish();
    
    double elapsedTime = device.timeBetween(start, end)/Ntests;
    
    size_t bytesIn  = 2*Nrun*sizeof(dfloat);
    size_t bytesOut = Nrun*sizeof(dfloat);
    size_t bytes = bytesIn + bytesOut;

    printf("BS2: " dlongFormat ", %4.4f, %1.2e, %1.2e, %4.1f ; dofs, elapsed, time per DOF, DOFs/time, BW (GB/s) \n",
	   Nrun, elapsedTime, elapsedTime/Nrun, ((dfloat) Nrun)/elapsedTime, bytes/(1e9*elapsedTime));
  }

  o_a.free();
  o_b.free();
  o_c.free();
}
