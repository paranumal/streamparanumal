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

#include "mesh.hpp"

namespace libp {

void mesh_t::SetupBoxTet3D(){

  //local grid physical sizes
  //Hard code to 2x2x2
  dfloat DIMX=2.0, DIMY=2.0, DIMZ=2.0;

  //number of local elements in each dimension
  dlong nx, ny, nz;
  settings.getSetting("BOX NX", nx);
  settings.getSetting("BOX NY", ny);
  settings.getSetting("BOX NZ", nz);

  // find a factorization size = size_x*size_y*size_z such that
  //  size_x>=size_y>=size_z are all 'close' to one another
  int size_x, size_y, size_z;
  Factor3(size, size_x, size_y, size_z);

  //determine (x,y,z) rank coordinates for this processes
  int rank_x=-1, rank_y=-1, rank_z=-1;
  RankDecomp3(size_x, size_y, size_z,
              rank_x, rank_y, rank_z,
              rank);

  //local grid physical sizes
  dfloat dimx = DIMX/size_x;
  dfloat dimy = DIMY/size_y;
  dfloat dimz = DIMZ/size_z;

  //bottom corner of physical domain
  dfloat X0 = -DIMX/2.0 + rank_x*dimx;
  dfloat Y0 = -DIMY/2.0 + rank_y*dimy;
  dfloat Z0 = -DIMZ/2.0 + rank_z*dimz;

  //global number of elements in each dimension
  hlong NX = size_x*nx;
  hlong NY = size_y*ny;
  hlong NZ = size_z*nz;

  //global number of nodes in each dimension
  hlong NnX = NX+1;
  hlong NnY = NY+1;
  hlong NnZ = NZ+1;

  // build an nx x ny x nz box grid
  Nnodes = NnX*NnY*NnZ; //global node count
  Nelements = 6*nx*ny*nz; //local element count (each cube divided into 6 tets)

  EToV.malloc(Nelements*Nverts);
  EX.malloc(Nelements*Nverts);
  EY.malloc(Nelements*Nverts);
  EZ.malloc(Nelements*Nverts);

  const dfloat dx = dimx/nx;
  const dfloat dy = dimy/ny;
  const dfloat dz = dimz/nz;

  #pragma omp parallel for collapse(3)
  for(int k=0;k<nz;++k){
    for(int j=0;j<ny;++j){
      for(int i=0;i<nx;++i){

        dlong e = 6*(i + j*nx + k*nx*ny);

        const hlong i0 = i+rank_x*nx;
        const hlong i1 = (i+1+rank_x*nx)%NnX;
        const hlong j0 = j+rank_y*ny;
        const hlong j1 = (j+1+rank_y*ny)%NnY;
        const hlong k0 = k+rank_z*nz;
        const hlong k1 = (k+1+rank_z*nz)%NnZ;

        dfloat x0 = X0 + dx*i;
        dfloat y0 = Y0 + dy*j;
        dfloat z0 = Z0 + dz*k;

        //tet 1 (0,3,2,7)
        EToV[e*Nverts+0] = i0 + j0*NnX + k0*NnX*NnY;
        EToV[e*Nverts+1] = i1 + j1*NnX + k0*NnX*NnY;
        EToV[e*Nverts+2] = i0 + j1*NnX + k0*NnX*NnY;
        EToV[e*Nverts+3] = i1 + j1*NnX + k1*NnX*NnY;

        EX[e*Nverts+0] = x0;    EY[e*Nverts+0] = y0;    EZ[e*Nverts+0] = z0;
        EX[e*Nverts+1] = x0+dx; EY[e*Nverts+1] = y0+dy; EZ[e*Nverts+1] = z0;
        EX[e*Nverts+2] = x0;    EY[e*Nverts+2] = y0+dy; EZ[e*Nverts+2] = z0;
        EX[e*Nverts+3] = x0+dx; EY[e*Nverts+3] = y0+dy; EZ[e*Nverts+3] = z0+dz;
        e++;

        //tet 2 (0,1,3,7)
        EToV[e*Nverts+0] = i0 + j0*NnX + k0*NnX*NnY;
        EToV[e*Nverts+1] = i1 + j0*NnX + k0*NnX*NnY;
        EToV[e*Nverts+2] = i1 + j1*NnX + k0*NnX*NnY;
        EToV[e*Nverts+3] = i1 + j1*NnX + k1*NnX*NnY;

        EX[e*Nverts+0] = x0;    EY[e*Nverts+0] = y0;    EZ[e*Nverts+0] = z0;
        EX[e*Nverts+1] = x0+dx; EY[e*Nverts+1] = y0;    EZ[e*Nverts+1] = z0;
        EX[e*Nverts+2] = x0+dx; EY[e*Nverts+2] = y0+dy; EZ[e*Nverts+2] = z0;
        EX[e*Nverts+3] = x0+dx; EY[e*Nverts+3] = y0+dy; EZ[e*Nverts+3] = z0+dz;
        e++;

        //tet 3 (0,2,6,7)
        EToV[e*Nverts+0] = i0 + j0*NnX + k0*NnX*NnY;
        EToV[e*Nverts+1] = i0 + j1*NnX + k0*NnX*NnY;
        EToV[e*Nverts+2] = i0 + j1*NnX + k1*NnX*NnY;
        EToV[e*Nverts+3] = i1 + j1*NnX + k1*NnX*NnY;

        EX[e*Nverts+0] = x0;    EY[e*Nverts+0] = y0;    EZ[e*Nverts+0] = z0;
        EX[e*Nverts+1] = x0;    EY[e*Nverts+1] = y0+dy; EZ[e*Nverts+1] = z0;
        EX[e*Nverts+2] = x0;    EY[e*Nverts+2] = y0+dy; EZ[e*Nverts+2] = z0+dz;
        EX[e*Nverts+3] = x0+dx; EY[e*Nverts+3] = y0+dy; EZ[e*Nverts+3] = z0+dz;
        e++;

        //tet 4 (0,6,4,7)
        EToV[e*Nverts+0] = i0 + j0*NnX + k0*NnX*NnY;
        EToV[e*Nverts+1] = i0 + j1*NnX + k1*NnX*NnY;
        EToV[e*Nverts+2] = i0 + j0*NnX + k1*NnX*NnY;
        EToV[e*Nverts+3] = i1 + j1*NnX + k1*NnX*NnY;

        EX[e*Nverts+0] = x0;    EY[e*Nverts+0] = y0;    EZ[e*Nverts+0] = z0;
        EX[e*Nverts+1] = x0;    EY[e*Nverts+1] = y0+dy; EZ[e*Nverts+1] = z0+dz;
        EX[e*Nverts+2] = x0;    EY[e*Nverts+2] = y0;    EZ[e*Nverts+2] = z0+dz;
        EX[e*Nverts+3] = x0+dx; EY[e*Nverts+3] = y0+dy; EZ[e*Nverts+3] = z0+dz;
        e++;

        //tet 5 (0,5,1,7)
        EToV[e*Nverts+0] = i0 + j0*NnX + k0*NnX*NnY;
        EToV[e*Nverts+1] = i1 + j0*NnX + k1*NnX*NnY;
        EToV[e*Nverts+2] = i1 + j0*NnX + k0*NnX*NnY;
        EToV[e*Nverts+3] = i1 + j1*NnX + k1*NnX*NnY;

        EX[e*Nverts+0] = x0;    EY[e*Nverts+0] = y0;    EZ[e*Nverts+0] = z0;
        EX[e*Nverts+1] = x0+dx; EY[e*Nverts+1] = y0;    EZ[e*Nverts+1] = z0+dz;
        EX[e*Nverts+2] = x0+dx; EY[e*Nverts+2] = y0;    EZ[e*Nverts+2] = z0;
        EX[e*Nverts+3] = x0+dx; EY[e*Nverts+3] = y0+dy; EZ[e*Nverts+3] = z0+dz;
        e++;

        //tet 6 (0,4,5,7)
        EToV[e*Nverts+0] = i0 + j0*NnX + k0*NnX*NnY;
        EToV[e*Nverts+1] = i0 + j0*NnX + k1*NnX*NnY;
        EToV[e*Nverts+2] = i1 + j0*NnX + k1*NnX*NnY;
        EToV[e*Nverts+3] = i1 + j1*NnX + k1*NnX*NnY;

        EX[e*Nverts+0] = x0;    EY[e*Nverts+0] = y0;    EZ[e*Nverts+0] = z0;
        EX[e*Nverts+1] = x0;    EY[e*Nverts+1] = y0;    EZ[e*Nverts+1] = z0+dz;
        EX[e*Nverts+2] = x0+dx; EY[e*Nverts+2] = y0;    EZ[e*Nverts+2] = z0+dz;
        EX[e*Nverts+3] = x0+dx; EY[e*Nverts+3] = y0+dy; EZ[e*Nverts+3] = z0+dz;
        e++;
      }
    }
  }
}

} //namespace libp
