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

#ifndef MESH_HPP
#define MESH_HPP 1

#include "core.hpp"
#include "settings.hpp"
#include "ogs.hpp"

namespace libp {

class mesh_t {
public:
  platform_t platform;
  settings_t settings;
  properties_t props;

  comm_t comm;
  int rank, size;

  /*************************/
  /* Element Data          */
  /*************************/
  int dim;
  int Nverts, Nfaces, NfaceVertices;
  int elementType;

  // indices of vertex nodes
  memory<int> vertexNodes;

  hlong Nnodes=0; //global number of element vertices
  memory<dfloat> EX; // coordinates of vertices for each element
  memory<dfloat> EY;
  memory<dfloat> EZ;

  dlong Nelements=0;       //local element count
  hlong NelementsGlobal=0; //global element count
  memory<hlong> EToV;      // element-to-vertex connectivity
  memory<dlong> EToE;      // element-to-element connectivity
  memory<int>   EToF;      // element-to-(local)face connectivity
  memory<int>   EToP;      // element-to-partition/process connectivity

  memory<dlong> VmapM;  // list of vertices on each face
  memory<dlong> VmapP;  // list of vertices that are paired with face vertices


  /*************************/
  /* FEM Space             */
  /*************************/
  int N=0, Np=0;             // N = Polynomial order and Np = Nodes per element
  memory<dfloat> r, s, t;    // coordinates of local nodes

  int Nq=0;            // N = Polynomial order, Nq=N+1
  memory<dfloat> gllz; // 1D GLL quadrature nodes
  memory<dfloat> gllw; // 1D GLL quadrature weights

  // face node info
  int Nfp=0;                // number of nodes per face
  memory<int> faceNodes;    // list of element reference interpolation nodes on element faces
  memory<dlong> vmapM;      // list of volume nodes that are face nodes
  memory<dlong> vmapP;      // list of volume nodes that are paired with face nodes
  memory<int> faceVertices; // list of mesh vertices on each face

  /*************************/
  /* MPI Data              */
  /*************************/
  // Halo exchange
  dlong totalHaloPairs=0;
  ogs::halo_t halo;

  // CG gather-scatter info
  ogs::ogs_t ogs;
  memory<hlong> globalIds;
  memory<dlong> GlobalToLocal;
  deviceMemory<dlong> o_GlobalToLocal;

  mesh_t()=default;
  mesh_t(platform_t& _platform,
         settings_t& _settings,
         comm_t _comm) {
    Setup(_platform, _settings, _comm);
  }

  ~mesh_t() = default;

  void Setup(platform_t& _platform,
             settings_t& _settings,
             comm_t _comm);

private:
  /*Element types*/
  static constexpr int TRIANGLES     =3;
  static constexpr int QUADRILATERALS=4;
  static constexpr int TETRAHEDRA    =6;
  static constexpr int HEXAHEDRA     =12;

  /*Set the type of mesh*/
  void SetElementType(const int eType);

  // box mesh
  void SetupBox() {
    switch (elementType) {
      case TRIANGLES:
        SetupBoxTri2D();
        break;
      case QUADRILATERALS:
        SetupBoxQuad2D();
        break;
      case TETRAHEDRA:
        SetupBoxTet3D();
        break;
      case HEXAHEDRA:
        SetupBoxHex3D();
        break;
    }
  }
  void SetupBoxTri2D();
  void SetupBoxQuad2D();
  void SetupBoxTet3D();
  void SetupBoxHex3D();

  // reference nodes and operators
  void ReferenceNodes() {
    switch (elementType) {
      case TRIANGLES:
        ReferenceNodesTri2D();
        break;
      case QUADRILATERALS:
        ReferenceNodesQuad2D();
        break;
      case TETRAHEDRA:
        ReferenceNodesTet3D();
        break;
      case HEXAHEDRA:
        ReferenceNodesHex3D();
        break;
    }
  }
  void ReferenceNodesTri2D();
  void ReferenceNodesQuad2D();
  void ReferenceNodesTet3D();
  void ReferenceNodesHex3D();


  /* build parallel face connectivity */
  void Connect();

  // face-vertex to face-vertex connection
  void ConnectFaceVertices();

  // face-node to face-node connection
  void ConnectFaceNodes();

  // setup halo region
  void HaloSetup();

  /* build global connectivity in parallel */
  void ConnectNodes();

  /* build global gather scatter ops */
  void GatherScatterSetup();

  // Set device properties
  void DeviceProperties();

  /*************************/
  /* FEM Space             */
  /*************************/
  //1D
  void Nodes1D(int N, dfloat r[]);
  void EquispacedNodes1D(int _N, dfloat _r[]);
  void OrthonormalBasis1D(dfloat a, int i, dfloat &P);
  void Vandermonde1D(int N, int Npoints, dfloat r[], dfloat V[]);
  void MassMatrix1D(int _Np, dfloat V[], dfloat MM[]);
  void InterpolationMatrix1D(int _N,
                             int NpointsIn, dfloat rIn[],
                             int NpointsOut, dfloat rOut[],
                             dfloat I[]);

  //Jacobi polynomial evaluation
  dfloat JacobiP(dfloat a, dfloat alpha, dfloat beta, int N);

  //Gauss-Legendre-Lobatto quadrature nodes
  void JacobiGLL(int N, dfloat x[], dfloat w[]=nullptr);

  //Nth order Gauss-Jacobi quadrature nodes and weights
  void JacobiGQ(dfloat alpha, dfloat beta, int N, dfloat x[], dfloat w[]);

  //Tris
  void NodesTri2D(int _N, dfloat _r[], dfloat _s[]);
  void FaceNodesTri2D(int _N, dfloat _r[], dfloat _s[], int _faceNodes[]);
  void VertexNodesTri2D(int _N, dfloat _r[], dfloat _s[], int _vertexNodes[]);
  void EquispacedNodesTri2D(int _N, dfloat _r[], dfloat _s[]);
  void FaceNodeMatchingTri2D(int _N, dfloat _r[], dfloat _s[],
                             int _faceNodes[], int R[]);

  void Warpfactor(int _N, int Npoints, dfloat _r[], dfloat _w[]);
  void WarpBlendTransformTri2D(int _N, int _Npoints, dfloat _r[], dfloat _s[], dfloat alphaIn=-1);

  //Quads
  void NodesQuad2D(int _N, dfloat _r[], dfloat _s[]);
  void FaceNodesQuad2D(int _N, dfloat _r[], dfloat _s[], int _faceNodes[]);
  void VertexNodesQuad2D(int _N, dfloat _r[], dfloat _s[], int _vertexNodes[]);
  void FaceNodeMatchingQuad2D(int _N, dfloat _r[], dfloat _s[],
                              int _faceNodes[], int R[]);

  //Tets
  void NodesTet3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[]);
  void FaceNodesTet3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[], int _faceNodes[]);
  void VertexNodesTet3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[], int _vertexNodes[]);
  void EquispacedNodesTet3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[]);
  void FaceNodeMatchingTet3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[],
                             int _faceNodes[], int R[]);
  void WarpShiftFace3D(int _N, int Npoints, dfloat alpha,
                             dfloat L1[], dfloat L2[], dfloat L3[],
                             dfloat w1[], dfloat w2[]);
  void WarpBlendTransformTet3D(int _N, int _Npoints, dfloat _r[], dfloat _s[], dfloat _t[], dfloat alphaIn=-1);


  //Hexs
  void NodesHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[]);
  void FaceNodesHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[],  int _faceNodes[]);
  void VertexNodesHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[], int _vertexNodes[]);
  void FaceNodeMatchingHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[],
                             int _faceNodes[], int R[]);
};

void meshAddSettings(settings_t& settings);
void meshReportSettings(settings_t& settings);

}

#endif

