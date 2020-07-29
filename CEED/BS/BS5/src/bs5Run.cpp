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

#include "bs5.hpp"

void bs5_t::Run(){

  //create arrays buffers
  
  int N = 0;
  int Nmin = 0, Nmax = 0, Nsamples = 1;
  int B = 0, Bmin = 0, Bmax = 0;
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
  
  int sc = 6*sizeof(dfloat);  // bytes moved per entry
  Nmin = Bmin/sc;
  Nmax = Bmax/sc;
  N = Nmax;
  
  occa::memory o_p  = device.malloc(N*sizeof(dfloat));
  occa::memory o_Ap = device.malloc(N*sizeof(dfloat));
  occa::memory o_x  = device.malloc(N*sizeof(dfloat));
  occa::memory o_r  = device.malloc(N*sizeof(dfloat));

  occa::memory o_rdotr = device.malloc(1*sizeof(dfloat));

  int maxNblock = (N+blockSize-1)/(blockSize);
  occa::memory o_tmp = device.malloc(maxNblock*sizeof(dfloat));
    
  const dfloat alpha = 1.0;

  // warm up
  {
    int Nwarm = 5;
    int Nblock = (N+blockSize-1)/(blockSize);
    Nblock = (Nblock>blockSize) ? blockSize : Nblock; //limit to blockSize entries
    
    for(int n=0;n<Nwarm;++n){ //warmup
      kernel1(Nblock, N, o_p, o_Ap, alpha, o_x, o_r, o_tmp); //partial reduction
      kernel2(Nblock, o_tmp, o_rdotr); //finish reduction
    }
  } 
  

  printf("%%%% BS id, dofs, elapsed, time per DOF, DOFs/time, BW (GB/s), Tgpu(C), Tjunction (C), Tmem (C), Freq. (GHz) \n");
  for(int samp=1;samp<=Nsamples;++samp){
    int Nrun = Nmin + (Nmax-Nmin)*((samp+1)*(samp+2)/(double)((Nsamples+1)*(Nsamples+2)));

    // rest gpu (do here to avoid clock drop after warm up)
    //    device.finish();
    //    usleep(1e6);
    
    int Nblock = (Nrun+blockSize-1)/(blockSize);
    Nblock = (Nblock>blockSize) ? blockSize : Nblock; //limit to blockSize entries
    
    device.finish();
    dfloat tic = MPI_Wtime();

    /* CGupdate Test */
    int Ntests = 50;
    for(int n=0;n<Ntests;++n){
      kernel1(Nblock, Nrun, o_p, o_Ap, alpha, o_x, o_r, o_tmp); //partial reduction
      kernel2(Nblock, o_tmp, o_rdotr); //finish reduction
    }
    
    //  occa::streamTag end = device.tagStream();
    device.finish();
    dfloat toc = MPI_Wtime();
    double elapsedTime = (toc-tic)/Ntests;
    
    size_t bytesIn  = 4*Nrun*sizeof(dfloat);
    size_t bytesOut = 2*Nrun*sizeof(dfloat);
    size_t bytes = bytesIn + bytesOut;
    
    void hipReadTemperatures(int dev, double *Tlist, double *freqList);
    double Tlist[3], freqList[3];
    hipReadTemperatures(9,Tlist, freqList); // hard coded for gpu
    
    printf("2, " dlongFormat ", %1.5le, %1.5le, %1.5le, %1.5le, %1.5le, %1.5le, %1.5le, %1.5le ;\n",
	   Nrun, (double)elapsedTime, (double)elapsedTime/Nrun, ((dfloat) Nrun)/elapsedTime, (double)(bytes/1.e9)/elapsedTime,
	   Tlist[0], Tlist[1], Tlist[2], freqList[0]);


    
    //    fflush(stdout);
    
  }

  o_tmp.free();
  o_p.free();
  o_Ap.free();
  o_r.free();
  o_x.free();
  o_rdotr.free();
}
