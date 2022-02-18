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

#include "mesh.hpp"

namespace libp {

void mesh_t::SetupBoxTri2D(){

  //local grid physical sizes
  //Hard code to 2x2
  dfloat DIMX=2.0, DIMY=2.0;

  //number of local elements in each dimension
  dlong nx, ny;
  settings.getSetting("BOX NX", nx);
  settings.getSetting("BOX NY", ny);

  // find a factorization size = size_x*size_y such that
  //  size_x>=size_y are 'close' to one another
  int size_x, size_y;
  Factor2(size, size_x, size_y);

  //determine (x,y) rank coordinates for this processes
  int rank_x=-1, rank_y=-1;
  RankDecomp2(size_x, size_y,
              rank_x, rank_y,
              rank);

  //local grid physical sizes
  dfloat dimx = DIMX/size_x;
  dfloat dimy = DIMY/size_y;

  //bottom corner of physical domain
  dfloat X0 = -DIMX/2.0 + rank_x*dimx;
  dfloat Y0 = -DIMY/2.0 + rank_y*dimy;

  //global number of elements in each dimension
  hlong NX = size_x*nx;
  hlong NY = size_y*ny;

  //global number of nodes in each dimension
  hlong NnX = NX+1;
  hlong NnY = NY+1;

  // build an nx x ny x nz box grid
  Nnodes = NnX*NnY; //global node count
  Nelements = 2*nx*ny; //local

  EToV.malloc(Nelements*Nverts);
  EX.malloc(Nelements*Nverts);
  EY.malloc(Nelements*Nverts);

  const dfloat dx = dimx/nx;
  const dfloat dy = dimy/ny;

  #pragma omp parallel for collapse(2)
  for(int j=0;j<ny;++j){
    for(int i=0;i<nx;++i){

      dlong e = 2*(i + j*nx);

      const hlong i0 = i+rank_x*nx;
      const hlong i1 = (i+1+rank_x*nx)%NnX;
      const hlong j0 = j+rank_y*ny;
      const hlong j1 = (j+1+rank_y*ny)%NnY;

      dfloat x0 = X0 + dx*i;
      dfloat y0 = Y0 + dy*j;

      EToV[e*Nverts+0] = i0 + j0*NnX;
      EToV[e*Nverts+1] = i1 + j0*NnX;
      EToV[e*Nverts+2] = i1 + j1*NnX;

      EX[e*Nverts+0] = x0;    EY[e*Nverts+0] = y0;
      EX[e*Nverts+1] = x0+dx; EY[e*Nverts+1] = y0;
      EX[e*Nverts+2] = x0+dx; EY[e*Nverts+2] = y0+dy;
      e++;

      EToV[e*Nverts+0] = i0 + j0*NnX;
      EToV[e*Nverts+1] = i1 + j1*NnX;
      EToV[e*Nverts+2] = i0 + j1*NnX;

      EX[e*Nverts+0] = x0;    EY[e*Nverts+0] = y0;
      EX[e*Nverts+1] = x0+dx; EY[e*Nverts+1] = y0+dy;
      EX[e*Nverts+2] = x0;    EY[e*Nverts+2] = y0+dy;
      e++;
    }
  }
}

} //namespace libp
