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

void bs0_t::Run(){

  //create arrays buffers
  int N = 1;

  occa::memory o_a = device.malloc(N*sizeof(dfloat));
  occa::memory o_b = device.malloc(N*sizeof(dfloat));

  int Nave = 10;
  int Nwarm = 100;
  for(int w=0;w<Nwarm;++w){
    Nave = (w<10) ? 1:10;
    
    double elapsed = 0;
    for(int a=0;a<Nave;++a){
      device.finish();
      double tic = MPI_Wtime();
      
      kernel(N, o_a, o_b);
      
      device.finish();
      double toc = MPI_Wtime();
      elapsed += toc-tic;
    }
    elapsed /= Nave;

    printf("%d %E %E %%%% warm up\n", w, elapsed, elapsed);

    usleep(10000);
  }


  for(int Ntests=1;Ntests<1024;Ntests+=(Ntests<2)?1:2){

    double elapsed = 0;
    for(int a=0;a<Nave;++a){

      device.finish();
      double tic = MPI_Wtime();
      
      for(int n=0;n<Ntests;++n){
	kernel(N, o_a, o_b);
      }
      
      device.finish();
      double toc = MPI_Wtime();
      
      elapsed += toc-tic;
    }
    
    elapsed /= (Ntests*Nave);

    printf("%5.4E %5.4e %%%% Ntests, time per test, elapsed\n", Ntests, elapsed/Ntests, elapsed);

    usleep(10000);
  }

  printf("];\n");


  
  o_a.free();
  o_b.free();
}
