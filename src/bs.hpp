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

#ifndef BS_HPP
#define BS_HPP 1

#include "mesh.hpp"
#include "timer.hpp"

using namespace libp;

class bsSettings_t: public settings_t {
public:
  bsSettings_t(const int argc, char** argv, comm_t _comm);
  void report();
};

class bs_t {
public:
  platform_t platform;
  settings_t settings;
  comm_t comm;
  mesh_t mesh;

  int problemNumber;
  int blockSize;

  dlong N, Ngather;
  hlong Nglobal, NgatherGlobal;
  deviceMemory<dfloat> o_a, o_b, o_scratch;

  kernel_t kernel;

  bs_t() = default;
  bs_t(platform_t &_platform,
       settings_t& _settings,
       comm_t& _comm) {
    Setup(_platform, _settings, _comm);
  }

  ~bs_t() = default;

  //setup
  void Setup(platform_t &_platform,
             settings_t& _settings,
             comm_t& _comm);

  void RunKernel(int NB);

  void Run();
  void RunSweep();
  void RunTuning();

  std::string testName();
  std::string fileName();
  std::string kernelName();


  void tuningParams(properties_t& kernelInfo);
  void BuildKernel(properties_t& props);
  void AllocBuffers(size_t B);

  size_t bytesMoved(int NB);
  size_t FLOPs(int NB);
  int bytesToArraySize(size_t B);
};


#endif

