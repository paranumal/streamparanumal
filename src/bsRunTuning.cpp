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
#include "parameters.hpp"

void bs_t::RunTuning(){

  if (problemNumber==0) {
    LIBP_FORCE_ABORT("Tuning mode not suported for BS0");
  }

  if (problemNumber==6 ||
      problemNumber==7 ||
      problemNumber==8) {
    LIBP_FORCE_ABORT("Tuning mode not yet suported for BS6, 7, or 8");
  }

  std::string name = "BS" + std::to_string(problemNumber) + ", " + testName();

  std::string parameterName;
  switch (problemNumber) {
    case 1: parameterName = "COPY_BLOCKSIZE"; break;
    case 2: parameterName = "AXPY_BLOCKSIZE"; break;
    case 3: parameterName = "NORM_BLOCKSIZE"; break;
    case 4: parameterName = "DOT_BLOCKSIZE"; break;
    case 5: parameterName = "AXPYDOT_BLOCKSIZE"; break;
  }

  size_t B = 0;
  settings.getSetting("BYTES", B);
  AllocBuffers(B);

  properties_t kernelInfo = platform.props(); //copy base occa properties

  int wavesize = 1;
  if (platform.device.mode() == "CUDA") wavesize = 32;
  if (platform.device.mode() == "HIP") wavesize = 64;

  double maxBW = 0.;
  int bestBlockSize = 0;

  for (int bs=wavesize;bs<=1024;bs+=wavesize) {

    properties_t props = kernelInfo;
    props["defines/"+parameterName] = bs;
    blockSize = bs;

    BuildKernel(props);

    int Nwarmup=20;
    int Ntests=20;

    for(int n=0;n<Nwarmup;++n){
      RunKernel(N);
    }

    timePoint_t start = PlatformTime(platform);
    for(int n=0;n<Ntests;++n){
      RunKernel(N);
    }
    timePoint_t end = PlatformTime(platform);
    double elapsed = ElapsedTime(start, end)/Ntests;

    size_t bytes = bytesMoved(N);

    if (comm.rank()==0){
      printf("%s: DOFs=" hlongFormat ", elapsed=%5.4e, BW (GB/s)=%6.2f, BW/rank (GB/s)=%6.2f, blockSize=%d\n",
             name.c_str(), Nglobal, elapsed,
             (double)(bytes/1.e9)/elapsed,
             (double)(bytes/1.e9)/(elapsed*comm.size()),
             bs);
    }

    double BW = (double)(bytes/1.e9)/elapsed;
    if (BW>maxBW) {
      maxBW = BW;
      bestBlockSize = bs;
    }
  }

  if (comm.rank()==0) {
    properties_t param;
    param["Name"] = "bs.okl";

    properties_t keys;
    keys["dfloat"] = (sizeof(dfloat)==4) ? "float" : "double";
    keys["mode"] = platform.device.mode();

    std::string arch = platform.device.arch();
    if (platform.device.mode()=="HIP") {
      arch = arch.substr(0,arch.find(":")); //For HIP mode, remove the stuff after the :
    }
    keys["arch"] = arch;

    properties_t props;
    props[parameterName] = bestBlockSize;

    param["keys"] = keys;
    param["props"] = props;

    std::cout << name << ":" << parameters_t::toString(param) << std::endl;
  }
}
