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

std::string bs_t::testName(){
  switch (problemNumber) {
    case 0: return "KLL          ";
    case 1: return "Copy         ";
    case 2: return "AXPY         ";
    case 3: return "Norm         ";
    case 4: return "DOT          ";
    case 5: return "AXPY+Norm    ";
    case 6: return "Gather       ";
    case 7: return "Scatter      ";
    case 8: return "GatherScatter";
    default: return "";
  }
}

std::string bs_t::fileName(){
  if (problemNumber==6 ||
      problemNumber==7 ||
      problemNumber==8) {
    return "";
  } else {
    return std::string(LIBP_DIR) + "/okl/bs" + std::to_string(problemNumber) + ".okl";
  }
}

std::string bs_t::kernelName(){
  if (problemNumber==6 ||
      problemNumber==7 ||
      problemNumber==8) {
    return "";
  } else {
    return "bs" + std::to_string(problemNumber);
  }
}

void bs_t::tuningParams(properties_t& kernelInfo) {

  parameters_t tuningParameters;

  std::string filename = std::string(LIBP_DIR) + "/json/bs.json";

  properties_t keys;
  keys["dfloat"] = (sizeof(dfloat)==4) ? "float" : "double";
  keys["mode"] = platform.device.mode();

  std::string arch = platform.device.arch();
  if (platform.device.mode()=="HIP") {
    arch = arch.substr(0,arch.find(":")); //For HIP mode, remove the stuff after the :
  }
  keys["arch"] = arch;

  std::string name = "bs.okl";

  if (settings.compareSetting("VERBOSE", "TRUE") && comm.rank()==0) {
    std::cout << "Loading Tuning Parameters, looking for match for Name:'" << name << "', keys:" << tuningParameters.toString(keys) << std::endl;
  }

  tuningParameters.load(filename, comm);

  properties_t matchProps = tuningParameters.findProperties(name, keys);

  if (settings.compareSetting("VERBOSE", "TRUE") && comm.rank()==0) {
    std::cout << "Found best match = " << tuningParameters.toString(matchProps) << std::endl;
  }

  kernelInfo["defines"] += matchProps["props"];

  /*Read blocksize from properties*/
  switch (problemNumber) {
    case 1: blockSize = static_cast<int>(matchProps["props/COPY_BLOCKSIZE"]); break;
    case 2: blockSize = static_cast<int>(matchProps["props/AXPY_BLOCKSIZE"]); break;
    case 3: blockSize = static_cast<int>(matchProps["props/NORM_BLOCKSIZE"]); break;
    case 4: blockSize = static_cast<int>(matchProps["props/DOT_BLOCKSIZE"]); break;
    case 5: blockSize = static_cast<int>(matchProps["props/AXPYDOT_BLOCKSIZE"]); break;
  }
}

size_t bs_t::bytesMoved(int NB) {

  switch (problemNumber) {
    case 0: return 0;
    case 1: return sizeof(dfloat) * NB * comm.size() * (1 + 1);
    case 2: return sizeof(dfloat) * NB * comm.size() * (2 + 1);
    case 3: return sizeof(dfloat) * NB * comm.size() * (1 + 0);
    case 4: return sizeof(dfloat) * NB * comm.size() * (2 + 0);
    case 5: return sizeof(dfloat) * NB * comm.size() * (2 + 1);
    case 6: return (sizeof(dfloat) * (NgatherGlobal+comm.size()) + //row starts
                    sizeof(dlong)  * Nglobal +           //localIds
                    sizeof(dfloat) * Nglobal +           //values
                    sizeof(dfloat) * NgatherGlobal);     //output
    case 7: return (sizeof(dfloat) * (NgatherGlobal+comm.size()) + //row starts
                    sizeof(dlong)  * Nglobal +           //localIds
                    sizeof(dfloat) * NgatherGlobal +     //values
                    sizeof(dfloat) * Nglobal);           //output
    case 8: return (sizeof(dfloat) * (NgatherGlobal+comm.size()) + //row starts
                    sizeof(dlong)  * Nglobal +           //localIds
                    sizeof(dfloat) * Nglobal +           //values
                    sizeof(dfloat) * Nglobal);           //output
    default: return 0;
  }
}

size_t bs_t::FLOPs(int NB) {
  switch (problemNumber) {
    case 0: return 0;
    case 1: return 0;
    case 2: return NB * comm.size() * 3;
    case 3: return NB * comm.size() * 2;
    case 4: return NB * comm.size() * 2;
    case 5: return NB * comm.size() * (3 + 2);
    case 6: return Nglobal;
    case 7: return 0;
    case 8: return Nglobal;
    default: return 0;
  }
}

int bs_t::bytesToArraySize(size_t B) {
  switch (problemNumber) {
    case 0: return 1;
    case 1: return B/(sizeof(dfloat) * (1 + 1));
    case 2: return B/(sizeof(dfloat) * (2 + 1));
    case 3: return B/(sizeof(dfloat) * (1 + 0));
    case 4: return B/(sizeof(dfloat) * (2 + 0));
    case 5: return B/(sizeof(dfloat) * (2 + 1));
    default: return 0;
  }
}
