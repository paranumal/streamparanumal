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

void bs_t::RunKernel(int NB){

  if (problemNumber==0) {

    kernel(NB, o_a, o_b); //KLL

  } else if (problemNumber==1) {

    kernel(NB, o_a, o_b); //Copy

  } else if (problemNumber==2) {

    dfloat alpha = 1.0, beta = 1.0;
    kernel(NB, alpha, o_a, beta, o_b); //AXPY

  } else if (problemNumber==3) {

    int Nblock = (NB+blockSize-1)/blockSize;
    Nblock = std::min(Nblock,blockSize); //limit to blockSize entries
    kernel(Nblock, NB, o_a, o_scratch); //Norm

  } else if (problemNumber==4) {

    int Nblock = (NB+blockSize-1)/blockSize;
    Nblock = std::min(Nblock,blockSize); //limit to blockSize entries
    kernel(Nblock, NB, o_a, o_b, o_scratch); //DOT

  } else if (problemNumber==5) {

    dfloat alpha = 1.0;
    int Nblock = (NB+blockSize-1)/blockSize;
    Nblock = std::min(Nblock,blockSize); //limit to blockSize entries
    kernel(Nblock, NB, alpha, o_a, o_b, o_scratch); //AXPY+Norm

  } else if (problemNumber==6) {

    mesh.ogs.Gather(o_b, o_a, 1, ogs::Add, ogs::Trans);

  } else if (problemNumber==7) {

    mesh.ogs.Scatter(o_b, o_a, 1, ogs::NoTrans);

  } else if (problemNumber==8) {

    mesh.ogs.GatherScatter(o_a, 1, ogs::Add, ogs::Sym);

  }
}
