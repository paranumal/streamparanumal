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

#ifndef MESH3D_HPP
#define MESH3D_HPP 1

#include "meshDefines3D.h"

class mesh3D: public mesh_t {
public:
  mesh3D(platform_t& _platform, settings_t& _settings, MPI_Comm _comm);

  // repartition elements in parallel
  void GeometricPartition();

  // serial face-node to face-node connection
  void ConnectFaceNodes();

  inline
  void ConnectFaceModes(int *faceModes, dfloat *V) {}; //not implemented yet

  // setup occa buffers
  virtual void OccaSetup();

  // print out mesh partition in parallel
  // void PrintVTU(const char *fileName);

  // dfloat ElementCharacteristicLength(dlong e);

  // dfloat MinCharacteristicLength();
};

class meshTri3D: public mesh3D {
public:
  meshTri3D(platform_t& _platform, settings_t& _settings, MPI_Comm _comm);
  void ParallelReader(const char *fileName);
  void SetupBox();
  // void SetupPmlBox();
  void ReferenceNodes(int N);
  void PhysicalNodes();
  // void GeometricFactors();
  // void SurfaceGeometricFactors();
  void OccaSetup();

  // void CubatureSetup();
  // void CubatureNodes();

  // void MassMatrixKernelSetup(int Nfields);
};

class meshQuad3D: public mesh3D {
public:
  meshQuad3D(platform_t& _platform, settings_t& _settings, MPI_Comm _comm);
  void ParallelReader(const char *fileName);
  void SetupBox();
  // void SetupPmlBox();
  void ReferenceNodes(int N);
  void PhysicalNodes();
  // void GeometricFactors();
  // void SurfaceGeometricFactors();
  void OccaSetup();

  // void CubatureSetup();
  // void CubatureNodes();

  // void MassMatrixKernelSetup(int Nfields);
};

class meshTet3D: public mesh3D {
public:
  meshTet3D(platform_t& _platform, settings_t& _settings, MPI_Comm _comm);
  void ParallelReader(const char *fileName);
  void SetupBox();
  // void SetupPmlBox();
  void ReferenceNodes(int N);
  void PhysicalNodes();
  // void GeometricFactors();
  // void SurfaceGeometricFactors();
  void OccaSetup();

  // void CubatureSetup();
  // void CubatureNodes();

  // void MassMatrixKernelSetup(int Nfields);
};

class meshHex3D: public mesh3D {
public:
  meshHex3D(platform_t& _platform, settings_t& _settings, MPI_Comm _comm);
  void ParallelReader(const char *fileName);
  void SetupBox();
  // void SetupPmlBox();
  void ReferenceNodes(int N);
  void PhysicalNodes();
  // void GeometricFactors();
  // void SurfaceGeometricFactors();
  void OccaSetup();

  // void CubatureSetup();
  // void CubatureNodes();

  // void MassMatrixKernelSetup(int Nfields);
};

#endif

