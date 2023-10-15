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

#include "bs9.hpp"

#define mymax(a,b) ( ((a)>(b)) ?  (a):(b) )

void bs9_t::Run(){

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

  int maxNreads = 16;
  int maxNwrites = 16;
  
  deviceMemory<dfloat> o_x = platform.malloc<dfloat>(Bmax/sizeof(dfloat));
  deviceMemory<dfloat> o_y = platform.malloc<dfloat>(Bmax/sizeof(dfloat));

  printf("%%%% knl, blockSize, Nreads Nwrites, T0, Rmax, maxBW, lastBW\n");
  printf("throughput = [\n");
  int minKernel = 1, maxKernel = 4;
  for(int knl=minKernel; knl<=maxKernel;++knl){
    for(int blockSize=16;blockSize<=1024;blockSize*=2){
      //    for(int powRead=0;powRead<=4;++powRead){
      //      for(int powWrite=0;powWrite<=4;++powWrite){
      //	int Nreads = pow(2,powRead);
      //	int Nwrites = pow(2,powWrite);
      for(int Nreads=1;Nreads<=maxNreads;++Nreads){
	for(int Nwrites=1;Nwrites<=maxNwrites;++Nwrites){
	  
      
	double A00 = 0, A01 = 0, A10 = 0, A11 = 0, b0 =0 , b1 =0;
      
	properties_t kernelInfo = platform.props(); //copy base occa properties

	kernelInfo["defines/" "p_blockSize"] = blockSize;
	kernelInfo["defines/" "p_Nreads"] = Nreads;
	kernelInfo["defines/" "p_Nwrites"] = Nwrites;

	kernelInfo["defines/p_knl"] = (int)knl;
      
#if 1
	occa::kernel readWriteKernel = platform.buildKernel(DBS9 "/okl/bs9.okl", "bs9", kernelInfo);
#else
	kernelInfo["okl/enabled"] = false;
	occa::kernel readWriteKernel = platform.buildKernel(DBS9 "/okl/bs9.cpp", "bs9", kernelInfo);
#endif

	// N = Bmax/(Nreads+Nwritea
      
	int sc = (Nreads+Nwrites)*sizeof(dfloat);  // bytes moved per entry
	int Nmin = Bmin/sc;
	int Nmax = Bmax/sc;
	int Nstep = (Bstep/sc > 0) ? Bstep/sc : 1;
      
	readWriteKernel.setRunDims((Nmin+blockSize-1)/blockSize, blockSize);
	    
	int Nwarm = 5;
	for(int n=0;n<Nwarm;++n){ //warmup
	  readWriteKernel(Nmin, o_x, o_y); //b = alpha*a + beta*b
	}
      
	if (B) {
	  //single test
	  int N = B/sc;
	  Nmin = N;
	  Nmax = N;
	  //	printf("BS9 = [");
	} else {
	  //sweep test
	  //	printf("%%AXPY [DOFs, elapsed, DOFs/s, BW (GB/s)]\n");
	  //	printf("BS9 = [\n");
	}

	double maxBW = 0, lastBW = 0;
	//test
	for(int N=Nmin;N<=Nmax;N+=Nstep){
	  readWriteKernel.setRunDims((N+blockSize-1)/blockSize, blockSize);

	  if(Bmax < Nreads*N*sizeof(dfloat) ||
	     Bmax < Nwrites*N*sizeof(dfloat)){
	    printf("Bmax = %lu, Nreads*N*sizeof(dfloat)=%lu, Nwrites*N*sizeof(dfloat)=%lu\n", Bmax, Nreads*N*sizeof(dfloat), Nwrites*N*sizeof(dfloat));
	  }

	  timePoint_t start = GlobalPlatformTime(platform);
	
	  /* AXPY Test */
	  int Ntests = 20;
	  for(int n=0;n<Ntests;++n){
	    readWriteKernel(N, o_x, o_y); //b = alpha*a + beta*b
	  }
	
	  timePoint_t end = GlobalPlatformTime(platform);
	  double elapsedTime = ElapsedTime(start, end)/Ntests;
	
	  size_t bytesIn  = Nreads*N*sizeof(dfloat);
	  size_t bytesOut = Nwrites*N*sizeof(dfloat);
	  size_t bytes = bytesIn + bytesOut;
	
	  //	printf("%d, %d, %d, %5.4e, %5.4e, %6.2f",
	  //	       N, Nreads, Nwrites, elapsedTime, N/elapsedTime, (double)(bytes/1.e9)/elapsedTime);
	  //	if (N<Nmax) printf(";\n");

	  // increment least squares
	  // [1 B]*[T0;1/Rmax] ~= [T] 
	  ++A00;
	  A01 += bytes/1.e9;
	  A10 += bytes/1.e9;
	  A11 += (bytes/1.e9)*(bytes/1.e9);
	  b0 += elapsedTime;
	  b1 += (bytes/1.e9)*elapsedTime;

	  maxBW = mymax(maxBW, (bytes/1.e9)/elapsedTime);
	  lastBW = (bytes/1.e9)/elapsedTime;
	}
	//      printf("]; %%AXPY [DOFs, Nreads, Nwrites, elapsed, DOFs/s, BW (GB/s)]\n");

	double J = A00*A11-A01*A01;
	double T0 = (A11*b0 - A01*b1)/J;
	dfloat Rmax = J/(-A10*b0 + A00*b1);
	printf("%d,%d,%d,%d,%lg,%d,%d,%d;\n", knl, blockSize, Nreads, Nwrites, T0, (int)Rmax, (int)maxBW, (int)lastBW);
      }
    }
  }
  }
  printf("];\n");
}
