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

#include "bs4.hpp"

bs4_t& bs4_t::Setup(occa::device& device, MPI_Comm& comm,
                    settings_t& settings, occa::properties& props) {

  bs4_t* bs4 = new bs4_t(device, comm, settings, props);

  // OCCA build stuff
  occa::properties kernelInfo = bs4->props; //copy base occa properties

  bs4->blockSize = 256;

  kernelInfo["defines/" "p_blockSize"] = bs4->blockSize;

  bs4->kernel1 = buildKernel(device, DBS4 "/okl/bs4.okl", "bs4_1", kernelInfo, comm);
  bs4->kernel2 = buildKernel(device, DBS4 "/okl/bs4.okl", "bs4_2", kernelInfo, comm);

  return *bs4;
}

bs4_t::~bs4_t() {
  kernel1.free();
  kernel2.free();
}
