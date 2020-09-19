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

void mesh_t::BoundarySetup(int Nfields){

  //make a node-wise bc flag using the gsop (prioritize Dirichlet boundaries over Neumann)
  mapB = (int *) calloc(Nelements*Np,sizeof(int));
  const int largeNumber = 1<<20;
  for (dlong e=0;e<Nelements;e++) {
    for (int n=0;n<Np;n++) mapB[n+e*Np] = largeNumber;
    for (int f=0;f<Nfaces;f++) {
      int bc = EToB[f+e*Nfaces];
      if (bc>0) {
        for (int n=0;n<Nfp;n++) {
          int fid = faceNodes[n+f*Nfp];
          mapB[fid+e*Np] = mymin(bc,mapB[fid+e*Np]);
        }
      }
    }
  }
  ogs->GatherScatter(mapB, ogs_int, ogs_min, ogs_sym);

  //use the bc flags to find masked ids
  Nmasked = 0;
  for (dlong n=0;n<Nelements*Np;n++) {
    if (mapB[n] == largeNumber) {//no boundary
      mapB[n] = 0.;
    } else if (mapB[n] == 1) {   //Dirichlet boundary
      Nmasked++;
    }
  }
  o_mapB = platform.malloc(Nelements*Np*sizeof(int), mapB);


  maskIds = (dlong *) calloc(Nmasked, sizeof(dlong));
  Nmasked =0; //reset
  for (dlong n=0;n<Nelements*Np;n++)
    if (mapB[n] == 1) maskIds[Nmasked++] = n;

  if (Nmasked) o_maskIds = platform.malloc(Nmasked*sizeof(dlong), maskIds);

  //make a masked version of the global id numbering
  maskedGlobalIds = (hlong *) calloc(Nelements*Np*Nfields,sizeof(hlong));
  for (dlong e=0;e<Nelements;e++) {
    for (int n=0;n<Np;n++) {
      hlong id = globalIds[e*Np+n];
      for (int f=0;f<Nfields;f++) {
        maskedGlobalIds[e*Np*Nfields+f*Np+n] = id*Nfields+f;
      }
    }
  }

  //mask
  for (dlong n=0;n<Nmasked;n++) {
    dlong id = maskIds[n];
    dlong e = id/Np;
    int m = id%Np;
    for (int f=0;f<Nfields;f++) {
      maskedGlobalIds[e*Np*Nfields+f*Np+m] = 0;
    }
  }

  //use the masked ids to make another gs handle (signed so the gather is defined)
  int verbose = 0;
  ogs_t::Unique(maskedGlobalIds, Nelements*Np*Nfields, comm);     //flag a unique node in every gather node
  ogsMasked = ogs_t::Setup(Nelements*Np*Nfields, maskedGlobalIds,
                           comm, verbose, platform);

  /* use the masked gs handle to define a global ordering */
  dlong Ntotal  = Np*Nelements*Nfields; // number of degrees of freedom on this rank (before gathering)
  hlong Ngather = ogsMasked->Ngather;     // number of degrees of freedom on this rank (after gathering)

  // build inverse degree vectors
  // used for the weight in linear solvers (used in C0)
  weight  = (dfloat*) calloc(Ntotal, sizeof(dfloat));
  weightG = (dfloat*) calloc(ogsMasked->Ngather, sizeof(dfloat));
  for(dlong n=0;n<Ntotal;++n) weight[n] = 1.0;

  ogsMasked->Gather(weightG, weight, ogs_dfloat, ogs_add, ogs_trans);
  for(dlong n=0;n<ogsMasked->Ngather;++n)
    if (weightG[n]) weightG[n] = 1./weightG[n];

  ogsMasked->Scatter(weight, weightG, ogs_dfloat, ogs_add, ogs_notrans);

  // o_weight  = platform.malloc(Ntotal*sizeof(dfloat), weight);
  // o_weightG = platform.malloc(ogsMasked->Ngather*sizeof(dfloat), weightG);

  // create a global numbering system
  hlong *newglobalIds = (hlong *) calloc(Ngather,sizeof(hlong));
  int   *owner     = (int *) calloc(Ngather,sizeof(int));

  // every gathered degree of freedom has its own global id
  hlong *globalStarts = (hlong*) calloc(size+1,sizeof(hlong));
  MPI_Allgather(&Ngather, 1, MPI_HLONG, globalStarts+1, 1, MPI_HLONG, comm);
  for(int rr=0;rr<size;++rr)
    globalStarts[rr+1] = globalStarts[rr] + globalStarts[rr+1];

  //use the offsets to set a consecutive global numbering
  for (dlong n =0;n<ogsMasked->Ngather;n++) {
    newglobalIds[n] = n + globalStarts[rank];
    owner[n] = rank;
  }

  //scatter this numbering to the original nodes
  maskedGlobalNumbering = (hlong *) calloc(Ntotal,sizeof(hlong));
  maskedGlobalOwners    = (int *)   calloc(Ntotal,sizeof(int));
  for (dlong n=0;n<Ntotal;n++) maskedGlobalNumbering[n] = -1;
  ogsMasked->Scatter(maskedGlobalNumbering, newglobalIds, ogs_hlong, ogs_add, ogs_notrans);
  ogsMasked->Scatter(maskedGlobalOwners, owner, ogs_int, ogs_add, ogs_notrans);

  free(newglobalIds); free(owner);

  /* Build global to local mapping */
  ogsMasked->GatheredHaloExchangeSetup();

  // mask
  // maskKernel = buildKernel(device, CEED_DIR "/core/okl/mask.okl",
  //                                    "mask", props, comm);
}