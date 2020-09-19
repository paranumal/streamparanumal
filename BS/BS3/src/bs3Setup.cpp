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

#include "bs3.hpp"

bs3_t& bs3_t::Setup(platform_t &platform, settings_t& settings) {

  bs3_t* bs3 = new bs3_t(platform, settings);

  // OCCA build stuff
  occa::properties kernelInfo = platform.props; //copy base occa properties

  bs3->blockSize = 256;

  kernelInfo["defines/" "p_blockSize"] = bs3->blockSize;

  bs3->kernel1 = platform.buildKernel(DBS3 "/okl/bs3.okl", "bs3_1", kernelInfo);
  bs3->kernel2 = platform.buildKernel(DBS3 "/okl/bs3.okl", "bs3_2", kernelInfo);

  return *bs3;
}

bs3_t::~bs3_t() {
  kernel1.free();
  kernel2.free();
}
