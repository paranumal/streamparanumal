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

#include "bs1.hpp"

bs1_t& bs1_t::Setup(occa::device& device, MPI_Comm& comm,
                    settings_t& settings, occa::properties& props) {

  bs1_t* bs1 = new bs1_t(device, comm, settings, props);

  // OCCA build stuff
  occa::properties kernelInfo = bs1->props; //copy base occa properties

  kernelInfo["defines/" "p_blockSize"] = (int)256;

  bs1->kernel = buildKernel(device, DBS1 "/okl/bs1.okl", "bs1", kernelInfo, comm);

  return *bs1;
}

bs1_t::~bs1_t() {
  kernel.free();
}
