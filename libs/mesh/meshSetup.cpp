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
#include "mesh/mesh2D.hpp"
#include "mesh/mesh3D.hpp"

mesh_t& mesh_t::Setup(platform_t& platform, settings_t& settings,
                      MPI_Comm comm){

  string eType;
  string fileName;
  int N, elementType = HEXAHEDRA;

  // settings.getSetting("MESH FILE", fileName);
  settings.getSetting("POLYNOMIAL DEGREE", N);
  settings.getSetting("ELEMENT TYPE", eType);
  // settings.getSetting("MESH DIMENSION", dim);

  mesh_t *mesh=nullptr;
  if (eType.compare("Tri")==0) {
    mesh = new meshTri2D(platform, settings, comm);
    elementType = TRIANGLES;
  } else if (eType.compare("Quad")==0) {
    mesh = new meshQuad2D(platform, settings, comm);
    elementType = QUADRILATERALS;
  } else if (eType.compare("Tet")==0) {
    mesh = new meshTet3D(platform, settings, comm);
    elementType = TETRAHEDRA;
  } else if (eType.compare("Hex")==0) {
    mesh = new meshHex3D(platform, settings, comm);
    elementType = HEXAHEDRA;
  }

  mesh->elementType = elementType;

  // mesh->ringHalo = NULL;

  // if (settings.compareSetting("MESH FILE","PMLBOX")) {
  //   //build a box mesh with a pml layer
  //   mesh->SetupPmlBox();
  // } else if (settings.compareSetting("MESH FILE","BOX")) {
  //   //build a box mesh
  //   mesh->SetupBox();
  // } else {
  //   // read chunk of elements from file
  //   mesh->ParallelReader(fileName.c_str());

  //   // partition elements using Morton ordering & parallel sort
  //   mesh->GeometricPartition();
  // }

  //build a box mesh
  mesh->SetupBox();

  // connect elements using parallel sort
  mesh->ParallelConnect();

  // print out connectivity statistics
  if (settings.compareSetting("VERBOSE", "TRUE"))
    mesh->PrintPartitionStatistics();

  // connect elements to boundary faces
  mesh->ConnectBoundary();

  // load reference (r,s) element nodes
  mesh->ReferenceNodes(N);

  // set up halo exchange info for MPI (do before connect face nodes)
  mesh->HaloSetup();

  // compute physical (x,y) locations of the element nodes
  mesh->PhysicalNodes();

  // compute geometric factors
  // mesh->GeometricFactors();

  // connect face nodes (find trace indices)
  mesh->ConnectFaceNodes();

  // compute surface geofacs
  // mesh->SurfaceGeometricFactors();

  // make a global indexing
  mesh->ParallelConnectNodes();

  // make an ogs operator and label local/global gather elements
  mesh->ParallelGatherScatterSetup();

  mesh->OccaSetup();

  return *mesh;
}
