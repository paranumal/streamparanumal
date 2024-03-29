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

// uniquely label each node with a global index, used for gatherScatter
void mesh_t::ConnectNodes(){

  // form continuous node numbering (local=>virtual gather)
  globalIds.malloc((totalHaloPairs+Nelements)*Np);

  hlong localNnodes = Np*Nelements;
  hlong gatherNodeStart = localNnodes;
  comm.Scan(localNnodes, gatherNodeStart);
  gatherNodeStart -= localNnodes;

  // use local numbering
  #pragma omp parallel for collapse(2)
  for(dlong e=0;e<Nelements;++e){
    for(int n=0;n<Np;++n){
      dlong id = e*Np+n;
      globalIds[id] = 1 + id + gatherNodeStart;
    }
  }

  hlong gatherChange = 1;

  // keep comparing numbers on positive and negative traces until convergence
  while(gatherChange>0){

    // reset change counter
    gatherChange = 0;

    // send halo data and recv into extension of buffer
    halo.Exchange(globalIds, Np);

    // compare trace nodes
    #pragma omp parallel for collapse(2)
    for(dlong e=0;e<Nelements;++e){
      for(int n=0;n<Nfp*Nfaces;++n){
        dlong id  = e*Nfp*Nfaces + n;
        dlong idM = vmapM[id];
        dlong idP = vmapP[id];
        hlong gidM = globalIds[idM];
        hlong gidP = globalIds[idP];

        if(gidP<gidM){
          ++gatherChange;
          globalIds[idM] = gidP;
        }
      }
    }

    // sum up changes
    comm.Allreduce(gatherChange);
  }
}

} //namespace libp
