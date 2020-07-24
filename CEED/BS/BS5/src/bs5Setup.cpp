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

#include "bs5.hpp"

bs5_t& bs5_t::Setup(occa::device& device, MPI_Comm& comm,
                    settings_t& settings, occa::properties& props) {

  bs5_t* bs5 = new bs5_t(device, comm, settings, props);

  // OCCA build stuff
  occa::properties kernelInfo = bs5->props; //copy base occa properties

  bs5->blockSize = 256;
  
  kernelInfo["defines/" "p_blockSize"] = bs5->blockSize;

  bs5->kernel1 = buildKernel(device, DBS5 "/okl/bs5.okl", "bs5_1", kernelInfo, comm);
  bs5->kernel2 = buildKernel(device, DBS5 "/okl/bs5.okl", "bs5_2", kernelInfo, comm);

  return *bs5;
}

bs5_t::~bs5_t() {
  kernel1.free();
  kernel2.free();
}
