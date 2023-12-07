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

#include "bs.hpp"

void bs_t::Setup(platform_t &_platform,
                 settings_t& _settings,
                 comm_t& _comm){
  platform = _platform;
  settings = _settings;
  comm = _comm;

  settings.getSetting("BENCHMARK PROBLEM", problemNumber);

  //Build mesh and ogs
  if (problemNumber==6 ||
      problemNumber==7 ||
      problemNumber==8) {
    mesh.Setup(platform, settings, comm);
  }
}

void bs_t::BuildKernel(properties_t& props) {

  std::string fName = fileName();

  if (fName.length()) {
    kernel = platform.buildKernel(fName, kernelName(), props);
  }
}

void bs_t::AllocBuffers(size_t B) {

  int blockSizeMax = 1024;

  if (problemNumber==0) {

    N = 1;
    o_a = platform.malloc<dfloat>(N);
    o_b = platform.malloc<dfloat>(N);

  } else if (problemNumber==1) {

    int sc = 2*sizeof(dfloat);  // bytes moved per entry
    N = B/sc;
    Nglobal = N;
    comm.Allreduce(Nglobal);
    o_a = platform.malloc<dfloat>(N);
    o_b = platform.malloc<dfloat>(N);

  } else if (problemNumber==2) {

    int sc = 3*sizeof(dfloat);  // bytes moved per entry
    N = B/sc;
    Nglobal = N;
    comm.Allreduce(Nglobal);
    o_a = platform.malloc<dfloat>(N);
    o_b = platform.malloc<dfloat>(N);

  } else if (problemNumber==3) {

    int sc = sizeof(dfloat);  // bytes moved per entry
    N = B/sc;
    Nglobal = N;
    comm.Allreduce(Nglobal);
    o_a = platform.malloc<dfloat>(N);
    o_scratch = platform.malloc<dfloat>(blockSizeMax);

  } else if (problemNumber==4) {

    int sc = 2*sizeof(dfloat);  // bytes moved per entry
    N = B/sc;
    Nglobal = N;
    comm.Allreduce(Nglobal);
    o_a = platform.malloc<dfloat>(N);
    o_b = platform.malloc<dfloat>(N);
    o_scratch = platform.malloc<dfloat>(blockSizeMax);

  } else if (problemNumber==5) {

    int sc = 3*sizeof(dfloat);  // bytes moved per entry
    N = B/sc;
    Nglobal = N;
    comm.Allreduce(Nglobal);
    o_a = platform.malloc<dfloat>(N);
    o_b = platform.malloc<dfloat>(N);
    o_scratch = platform.malloc<dfloat>(blockSizeMax);

  } else if (problemNumber==6) {

    N = mesh.Np*mesh.Nelements;
    Ngather = mesh.ogs.Ngather;

    Nglobal = N;
    comm.Allreduce(Nglobal);
    NgatherGlobal = Ngather;
    comm.Allreduce(NgatherGlobal);

    o_a = platform.malloc<dfloat>(N);
    o_b = platform.malloc<dfloat>(Ngather);

  } else if (problemNumber==7) {

    N = mesh.Np*mesh.Nelements;
    Ngather = mesh.ogs.Ngather;

    Nglobal = N;
    comm.Allreduce(Nglobal);
    NgatherGlobal = Ngather;
    comm.Allreduce(NgatherGlobal);

    o_a = platform.malloc<dfloat>(Ngather);
    o_b = platform.malloc<dfloat>(N);

  } else if (problemNumber==8) {

    N = mesh.Np*mesh.Nelements;
    Ngather = mesh.ogs.Ngather;

    Nglobal = N;
    comm.Allreduce(Nglobal);
    NgatherGlobal = Ngather;
    comm.Allreduce(NgatherGlobal);

    o_a = platform.malloc<dfloat>(N);
    o_b = platform.malloc<dfloat>(N);

  }

}
