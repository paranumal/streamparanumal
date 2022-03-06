/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

#ifndef CORE_HPP
#define CORE_HPP

#include "utils.hpp"
#include "memory.hpp"
#include "comm.hpp"

namespace libp {

// find a factorization n = nx*ny such that
//  nx>=ny are 'close' to one another
void Factor2(const int n, int &nx, int &ny);

void RankDecomp2(int  size_x, int  size_y,
                 int &rank_x, int &rank_y,
                 const int rank);

// find a factorization n = nx*ny*nz such that
//  nx>=ny>=nz are all 'close' to one another
void Factor3(const int n, int &nx, int &ny, int &nz);

void RankDecomp3(int  size_x, int  size_y, int  size_z,
                 int &rank_x, int &rank_y, int &rank_z,
                 const int rank);

void matrixRightSolve(int NrowsA, int NcolsA, double *A, int NrowsB, int NcolsB, double *B, double *C);
void matrixRightSolve(int NrowsA, int NcolsA, float *A, int NrowsB, int NcolsB, float *B, float *C);

void matrixEigenVectors(int N, double *A, double *VR, double *WR, double *WI);
void matrixEigenVectors(int N, float *A, float *VR, float *WR, float *WI);

void matrixEigenValues(int N, double *A, double *WR, double *WI);
void matrixEigenValues(int N, float *A, float *WR, float *WI);

void matrixInverse(int N, double *A);
void matrixInverse(int N, float *A);

void matrixTranspose(const int M, const int N,
                     const double  *A, const int LDA,
                           double *AT, const int LDAT);
void matrixTranspose(const int M, const int N,
                     const float  *A, const int LDA,
                           float *AT, const int LDAT);

void matrixTranspose(const int M, const int N,
                     const int  *A, const int LDA,
                           int *AT, const int LDAT);
void matrixTranspose(const int M, const int N,
                     const long long int  *A, const int LDA,
                           long long int *AT, const int LDAT);

} //namespace libp

#endif
