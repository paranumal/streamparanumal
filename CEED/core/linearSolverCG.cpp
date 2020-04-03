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

#include "linearSolver.hpp"

#define CG_BLOCKSIZE 512

cg::cg(solver_t& _solver):
  linearSolver_t(_solver) {};

cg::~cg() {
  updateCGKernel.free();
}

void cg::Init(int _weighted, occa::memory& o_weight,
             dlong _N, dlong Nhalo) {

  N = _N;
  dlong Ntotal = N + Nhalo;

  /*aux variables */
  dfloat *dummy = (dfloat *) calloc(Ntotal,sizeof(dfloat)); //need this to avoid uninitialized memory warnings
  o_p  = device.malloc(Ntotal*sizeof(dfloat),dummy);
  o_Ax = device.malloc(Ntotal*sizeof(dfloat),dummy);
  o_Ap = device.malloc(Ntotal*sizeof(dfloat),dummy);
  free(dummy);

  weighted = _weighted;
  o_w = o_weight;

  //pinned tmp buffer for reductions
  occa::properties mprops;
  mprops["mapped"] = true;
  h_tmprdotr = device.malloc(CG_BLOCKSIZE*sizeof(dfloat), mprops);
  tmprdotr = (dfloat*) h_tmprdotr.ptr(mprops);
  o_tmprdotr = device.malloc(CG_BLOCKSIZE*sizeof(dfloat));

  /* build kernels */
  occa::properties kernelInfo = props; //copy base properties

  //add defines
  kernelInfo["defines/" "p_blockSize"] = (int)CG_BLOCKSIZE;

  // combined CG update and r.r kernel
  updateCGKernel = buildKernel(device,
                                CEED_DIR "/core/okl/linearSolverUpdateCG.okl",
                                "updateCG", kernelInfo, comm);
}

int cg::Solve(solver_t& solver,
               occa::memory &o_x, occa::memory &o_r,
               const dfloat tol, const int MAXIT, const int verbose) {

  int rank = mesh.rank;

  // register scalars
  dfloat rdotr1 = 0.0;
  dfloat rdotr2 = 0.0;
  dfloat alpha = 0.0, beta = 0.0, pAp = 0.0;
  dfloat rdotr = 0.0;

  // compute A*x
  solver.Operator(o_x, o_Ax);

  // subtract r = r - A*x
  linAlg.axpy(N, -1.f, o_Ax, 1.f, o_r);

  if (weighted)
    rdotr = linAlg.weightedNorm2(N, o_w, o_r, comm);
  else
    rdotr = linAlg.norm2(N, o_r, comm);
  rdotr = rdotr*rdotr;

  dfloat TOL = mymax(tol*tol*rdotr,tol*tol);

  if (verbose&&(rank==0))
    printf("CG: initial res norm %12.12f \n", sqrt(rdotr));

  int iter;
  for(iter=0;iter<MAXIT;++iter){

    //exit if tolerance is reached
    if(rdotr<=TOL) break;

    // r.r
    rdotr2 = rdotr1;
    rdotr1 = rdotr; //computed in UpdateCG

    beta = (iter==0) ? 0.0 : rdotr1/rdotr2;

    // p = r + beta*p
    linAlg.axpy(N, 1.f, o_r, beta, o_p);

    // A*p
    solver.Operator(o_p, o_Ap);

    // p.Ap
    if (weighted)
      pAp =  linAlg.weightedInnerProd(N, o_w, o_p, o_Ap, comm);
    else
      pAp =  linAlg.innerProd(N, o_p, o_Ap, comm);

    alpha = rdotr1/pAp;

    //  x <= x + alpha*p
    //  r <= r - alpha*A*p
    //  dot(r,r)
    rdotr = UpdateCG(alpha, o_x, o_r);

    if (verbose&&(rank==0)) {
      if(rdotr<0)
        printf("WARNING CG: rdotr = %17.15lf\n", rdotr);

      printf("CG: it %d, r norm %12.12le, alpha = %le \n", iter+1, sqrt(rdotr), alpha);
    }
  }

  return iter;
}

dfloat cg::UpdateCG(const dfloat alpha, occa::memory &o_x, occa::memory &o_r){

  // x <= x + alpha*p
  // r <= r - alpha*A*p
  // dot(r,r)
  int Nblocks = (N+CG_BLOCKSIZE-1)/CG_BLOCKSIZE;
  Nblocks = (Nblocks>CG_BLOCKSIZE) ? CG_BLOCKSIZE : Nblocks; //limit to CG_BLOCKSIZE entries

  updateCGKernel(N, Nblocks, weighted, o_w, o_p, o_Ap, alpha, o_x, o_r, o_tmprdotr);

  o_tmprdotr.copyTo(tmprdotr, Nblocks*sizeof(dfloat));

  dfloat rdotr1 = 0;
  for(int n=0;n<Nblocks;++n)
    rdotr1 += tmprdotr[n];

  dfloat globalrdotr1 = 0;
  MPI_Allreduce(&rdotr1, &globalrdotr1, 1, MPI_DFLOAT, MPI_SUM, comm);
  return globalrdotr1;
}