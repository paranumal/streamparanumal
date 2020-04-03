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

#ifndef MESH_HPP
#define MESH_HPP 1

#include <unistd.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <occa.hpp>

#include "types.h"
#include "utils.hpp"
#include "ogs.hpp"
#include "settings.hpp"

#define TRIANGLES 3
#define QUADRILATERALS 4
#define TETRAHEDRA 6
#define HEXAHEDRA 12

class mesh_t {
public:
  occa::device& device;
  MPI_Comm& comm;
  settings_t& settings;
  occa::properties& props;

  int rank, size; // MPI rank and size (process count)

  int dim;
  int Nverts, Nfaces, NfaceVertices;

  int elementType;

  hlong Nnodes; //global number of element vertices
  dfloat *EX; // coordinates of vertices for each element
  dfloat *EY;
  dfloat *EZ;

  dlong Nelements;       //local element count
  hlong NelementsGlobal; //global element count
  hlong *EToV; // element-to-vertex connectivity
  dlong *EToE; // element-to-element connectivity
  int   *EToF; // element-to-(local)face connectivity
  int   *EToP; // element-to-partition/process connectivity
  int   *EToB; // element-to-boundary condition type

  hlong *elementInfo; //type of element

  // boundary faces
  hlong NboundaryFaces; // number of boundary faces
  hlong *boundaryInfo; // list of boundary faces (type, vertex-1, vertex-2, vertex-3)

  // MPI halo exchange info
  halo_t *halo;            // halo exchange pointer
  halo_t *ringHalo;        // ring halo exchange pointer
  dlong NinternalElements; // number of elements that can update without halo exchange
  dlong NhaloElements;     // number of elements that cannot update without halo exchange
  dlong  totalHaloPairs;   // number of elements to be received in halo exchange
  dlong  totalRingElements;// number of elements to be received in ring halo exchange
  dlong *internalElementIds;  // list of elements that can update without halo exchange
  dlong *haloElementIds;      // list of elements to be sent in halo exchange
  occa::memory o_internalElementIds;  // list of elements that can update without halo exchange
  occa::memory o_haloElementIds;      // list of elements to be sent in halo exchange

  // CG gather-scatter info
  ogs_t *ogs;              //occa gs pointer
  hlong *globalIds;

  // list of elements that are needed for global gather-scatter
  dlong NglobalGatherElements;
  dlong *globalGatherElementList;
  occa::memory o_globalGatherElementList;

  // list of elements that are not needed for global gather-scatter
  dlong NlocalGatherElements;
  dlong *localGatherElementList;
  occa::memory o_localGatherElementList;

  //C0-FEM mask data
  ogs_t *ogsMasked;
  int *mapB;      // boundary flag of face nodes

  dlong Nmasked;
  dlong *maskIds;
  hlong *maskedGlobalIds;
  hlong *maskedGlobalNumbering;
  int   *maskedGlobalOwners;

  occa::memory o_maskIds;
  occa::memory o_mapB;

  dfloat *weight, *weightG;
  occa::memory o_weight, o_weightG;
  occa::kernel maskKernel;

  // volumeGeometricFactors;
  dlong Nvgeo;
  dfloat *vgeo;

  // second order volume geometric factors
  dlong Nggeo;
  dfloat *ggeo;

  // volume node info
  int N, Np;
  dfloat *r, *s, *t;    // coordinates of local nodes
  dfloat *Dr, *Ds, *Dt; // collocation differentiation matrices
  dfloat *Dmatrices;
  dfloat *MM, *invMM;           // reference mass matrix
  dfloat *Srr,*Srs, *Srt; //element stiffness matrices
  dfloat *Ssr,*Sss, *Sst;
  dfloat *Str,*Sts, *Stt;
  dfloat *Smatrices;
  dfloat *x, *y, *z;    // coordinates of physical nodes

  dfloat sphereRadius;  // for Quad3D


  // indices of vertex nodes
  int *vertexNodes;

  // quad specific quantity
  int Nq, NqP, NpP;

  dfloat *D; // 1D differentiation matrix (for tensor-product)
  dfloat *gllz; // 1D GLL quadrature nodes
  dfloat *gllw; // 1D GLL quadrature weights

  int gjNq;
  dfloat *gjr,*gjw; // 1D nodes and weights for Gauss Jacobi quadature
  dfloat *gjI,*gjD; // 1D GLL to Gauss node interpolation and differentiation matrices
  dfloat *gjD2;     // 1D GJ to GJ node differentiation

  /* GeoData for affine mapped elements */
  dfloat *EXYZ;  // element vertices for reconstructing geofacs
  dfloat *gllzw; // GLL nodes and weights
  dfloat *ggeoNoJW;
  occa::memory o_EXYZ;
  occa::memory o_gllzw;
  occa::memory o_ggeoNoJW;

  // face node info
  int Nfp;        // number of nodes per face
  int *faceNodes; // list of element reference interpolation nodes on element faces
  dlong *vmapM;     // list of volume nodes that are face nodes
  dlong *vmapP;     // list of volume nodes that are paired with face nodes
  dlong *mapP;     // list of surface nodes that are paired with -ve surface  nodes
  int *faceVertices; // list of mesh vertices on each face

  dfloat *LIFT; // lift matrix
  dfloat *FMM;  // Face Mass Matrix
  dfloat *sMT; // surface mass (MM*LIFT)^T

  dlong   Nsgeo;
  dfloat *sgeo;

  // cubature
  int cubN, cubNp, cubNfp, cubNq;
  dfloat *cubr, *cubs, *cubt, *cubw; // coordinates and weights of local cubature nodes
  dfloat *cubx, *cuby, *cubz;    // coordinates of physical nodes
  dfloat *cubInterp; // interpolate from W&B to cubature nodes
  dfloat *cubProject; // projection matrix from cubature nodes to W&B nodes
  dfloat *cubD;       // 1D differentiation matrix
  dfloat *cubDW;     // 1D weak differentiation matrix
  dfloat *cubDrW;    // 'r' weak differentiation matrix
  dfloat *cubDsW;    // 's' weak differentiation matrix
  dfloat *cubDtW;    // 't' weak differentiation matrix
  dfloat *cubDWmatrices;

  dfloat *cubvgeo;  //volume geometric data at cubature points
  dfloat *cubsgeo;  //surface geometric data at cubature points
  dfloat *cubggeo;  //second type volume geometric data at cubature points


  // surface integration node info
  int    intNfp;    // number of integration nodes on each face
  dfloat *intInterp; // interp from surface node to integration nodes
  dfloat *intLIFT;   // lift from surface integration nodes to W&B volume nodes
  dfloat *intx, *inty, *intz; // coordinates of suface integration nodes

  //degree raising and lowering interpolation matrices
  dfloat *interpRaise;
  dfloat *interpLower;

  //pml lists
  dlong NnonPmlElements;
  dlong NpmlElements;

  dlong *pmlElements;
  dlong *nonPmlElements;
  dlong *pmlIds;

  //multirate lists
  int mrNlevels;
  int *mrLevel;
  dlong *mrNelements, *mrInterfaceNelements;
  dlong **mrElements, **mrInterfaceElements;

  //multirate pml lists
  dlong *mrNnonPmlElements, *mrNpmlElements;
  dlong **mrPmlElements, **mrNonPmlElements;
  dlong **mrPmlIds;

  // ploting info for generating field vtu
  int    plotNverts;    // number of vertices for each plot element
  int    plotNp;        // number of plot nodes per element
  int    plotNelements; // number of "plot elements" per element
  int   *plotEToV;      // triangulation of plot nodes
  dfloat *plotR, *plotS, *plotT; // coordinates of plot nodes in reference element
  dfloat *plotInterp;    // warp & blend to plot node interpolation matrix

  int *contourEToV;
  dfloat *contourVX, *contourVY, *contourVZ;
  dfloat *contourInterp, *contourInterp1, *contourFilter;

  //SEMFEM data
  int NpFEM, NelFEM;
  int *FEMEToV;
  dfloat *rFEM, *sFEM, *tFEM;
  dfloat *SEMFEMInterp;

  occa::memory o_SEMFEMInterp;
  occa::memory o_SEMFEMAnterp;


  // occa stuff
  occa::stream defaultStream;

  occa::memory o_Dr, o_Ds, o_Dt, o_LIFT, o_MM;
  occa::memory o_DrT, o_DsT, o_DtT, o_LIFTT;
  occa::memory o_Dmatrices;
  occa::memory o_FMMT;
  occa::memory o_sMT;

  occa::memory o_D; // tensor product differentiation matrix (for Hexes)
  occa::memory o_SrrT, o_SrsT, o_SrtT; //element stiffness matrices
  occa::memory o_SsrT, o_SssT, o_SstT;
  occa::memory o_StrT, o_StsT, o_SttT;
  occa::memory o_Srr, o_Srs, o_Srt, o_Sss, o_Sst, o_Stt; // for char4-based kernels
  occa::memory o_Smatrices;

  occa::memory o_vgeo, o_sgeo;
  occa::memory o_vmapM, o_vmapP, o_mapP;

  occa::memory o_rmapP;

  occa::memory o_EToE, o_EToF, o_EToB, o_x, o_y, o_z;

  // cubature
  occa::memory o_intLIFTT, o_intInterpT, o_intx, o_inty, o_intz;
  occa::memory o_cubDWT, o_cubD;
  occa::memory o_cubDrWT, o_cubDsWT, o_cubDtWT;
  occa::memory o_cubDWmatrices;
  occa::memory o_cubInterpT, o_cubProjectT;
  occa::memory o_cubInterp;
  occa::memory o_invMc; // for comparison: inverses of weighted mass matrices

  occa::memory o_cubvgeo, o_cubsgeo, o_cubggeo;

  //pml lists
  occa::memory o_pmlElements;
  occa::memory o_nonPmlElements;
  occa::memory o_pmlIds;

  //multirate lists
  occa::memory o_mrLevel;
  occa::memory o_mrNelements, o_mrInterfaceNelements;
  occa::memory *o_mrElements, *o_mrInterfaceElements;

  //multirate pml lists
  occa::memory *o_mrPmlElements, *o_mrNonPmlElements;
  occa::memory *o_mrPmlIds;

  occa::memory o_ggeo; // second order geometric factors
  occa::memory o_projectL2; // local weights for projection.

  mesh_t() = delete;
  mesh_t(occa::device& device, MPI_Comm& comm,
         settings_t& settings, occa::properties& props);

  virtual ~mesh_t();

  // generic mesh setup
  static mesh_t& Setup(occa::device& device, MPI_Comm& comm,
                       settings_t& settings, occa::properties& props);

  // box mesh
  virtual void SetupBox() = 0;

  /* build parallel face connectivity */
  void ParallelConnect();
  void Connect();

  // build element-boundary connectivity
  void ConnectBoundary();

  virtual void ReferenceNodes(int N) = 0;

  /* compute x,y,z coordinates of each node */
  virtual void PhysicalNodes() = 0;

  // compute geometric factors for local to physical map
  virtual void GeometricFactors() = 0;

  virtual void SurfaceGeometricFactors() = 0;

  // serial face-node to face-node connection
  virtual void ConnectFaceNodes() = 0;

  // setup halo region
  void HaloSetup();

  // setup trace halo
  halo_t* HaloTraceSetup(int Nfields);

  /* build global connectivity in parallel */
  void ParallelConnectNodes();

  /* build global gather scatter ops */
  void ParallelGatherScatterSetup();

  virtual void OccaSetup();

  virtual void CubatureSetup()=0;

  void BoundarySetup();

  // print out parallel partition i
  void PrintPartitionStatistics();

  virtual void PlotFields(dfloat* Q, int Nfields, char *fileName)=0;

protected:
  //1D
  void Nodes1D(int N, dfloat *r);
  void OrthonormalBasis1D(dfloat a, int i, dfloat *P);
  void GradOrthonormalBasis1D(dfloat a, int i, dfloat *Pr);
  void Vandermonde1D(int N, int Npoints, dfloat *r, dfloat *V);
  void GradVandermonde1D(int N, int Npoints, dfloat *r, dfloat *Vr);

  void MassMatrix1D(int _Np, dfloat *V, dfloat *MM);
  void Dmatrix1D(int N, int Npoints, dfloat *r, dfloat *Dr);
  void InterpolationMatrix1D(int _N,int NpointsIn, dfloat *rIn, int NpointsOut, dfloat *rOut, dfloat *I);

  //Jacobi polynomial evaluation
  dfloat JacobiP(dfloat a, dfloat alpha, dfloat beta, int N);
  dfloat GradJacobiP(dfloat a, dfloat alpha, dfloat beta, int N);

  //Gauss-Legendre-Lobatto quadrature nodes
  void JacobiGLL(int N, dfloat *x, dfloat *w=NULL);

  //Nth order Gauss-Jacobi quadrature nodes and weights
  void JacobiGQ(dfloat alpha, dfloat beta, int N, dfloat *x, dfloat *w);

  //Quads
  void NodesQuad2D(int _N, dfloat *_r, dfloat *_s);
  void FaceNodesQuad2D(int _N, dfloat *_r, dfloat *_s, int *_faceNodes);
  void VertexNodesQuad2D(int _N, dfloat *_r, dfloat *_s, int *_vertexNodes);
  void EquispacedNodesQuad2D(int _N, dfloat *_r, dfloat *_s);
  void EquispacedEToVQuad2D(int _N, int *_EToV);
  void OrthonormalBasisQuad2D(dfloat a, dfloat b, int i, int j, dfloat *P);
  void GradOrthonormalBasisQuad2D(dfloat a, dfloat b, int i, int j, dfloat *Pr, dfloat *Ps);
  void VandermondeQuad2D(int N, int Npoints, dfloat *r, dfloat *s, dfloat *V);
  void GradVandermondeQuad2D(int N, int Npoints, dfloat *r, dfloat *s, dfloat *Vr, dfloat *Vs);
  void MassMatrixQuad2D(int _Np, dfloat *V, dfloat *_MM);
  void DmatrixQuad2D(int _N, int Npoints, dfloat *_r, dfloat *_s,
                                          dfloat *_Dr, dfloat *_Ds);
  void InterpolationMatrixQuad2D(int _N,
                               int NpointsIn, dfloat *rIn, dfloat *sIn,
                               int NpointsOut, dfloat *rOut, dfloat *sOut,
                               dfloat *I);

  //Hexs
  void NodesHex3D(int _N, dfloat *_r, dfloat *_s, dfloat *_t);
  void FaceNodesHex3D(int _N, dfloat *_r, dfloat *_s, dfloat *_t,  int *_faceNodes);
  void VertexNodesHex3D(int _N, dfloat *_r, dfloat *_s, dfloat *_t, int *_vertexNodes);
  void EquispacedNodesHex3D(int _N, dfloat *_r, dfloat *_s, dfloat *_t);
  void EquispacedEToVHex3D(int _N, int *_EToV);
  void OrthonormalBasisHex3D(dfloat a, dfloat b, dfloat c, int i, int j, int k, dfloat *P);
  void GradOrthonormalBasisHex3D(dfloat a, dfloat b, dfloat c, int i, int j, int k, dfloat *Pr, dfloat *Ps, dfloat *Pt);
  void VandermondeHex3D(int N, int Npoints, dfloat *r, dfloat *s, dfloat *t, dfloat *V);
  void GradVandermondeHex3D(int N, int Npoints, dfloat *r, dfloat *s, dfloat *t,
                                                dfloat *Vr, dfloat *Vs, dfloat *Vt);
  void MassMatrixHex3D(int _Np, dfloat *V, dfloat *_MM);
  void DmatrixHex3D(int _N, int Npoints, dfloat *_r, dfloat *_s, dfloat *_t,
                                          dfloat *_Dr, dfloat *_Ds, dfloat *_Dt);
  void InterpolationMatrixHex3D(int _N,
                               int NpointsIn, dfloat *rIn, dfloat *sIn, dfloat *tIn,
                               int NpointsOut, dfloat *rOut, dfloat *sOut, dfloat *tOut,
                               dfloat *I);
};

void meshAddSettings(settings_t& settings);
void meshReportSettings(settings_t& settings);

#endif

