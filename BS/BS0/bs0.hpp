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

#ifndef BS0_HPP
#define BS0_HPP 1

#include "core.hpp"

#define DBS0 CEED_DIR"/BS/BS0/"

class bs0Settings_t: public settings_t {
public:
  bs0Settings_t(const int argc, char** argv, MPI_Comm& _comm);
  void report();
};

class bs0_t {
public:
  occa::device& device;
  MPI_Comm& comm;
  settings_t& settings;
  occa::properties& props;

  occa::kernel kernel;

  bs0_t() = delete;
  bs0_t(occa::device& _device, MPI_Comm& _comm,
        settings_t& _settings, occa::properties& _props):
    device(_device), comm(_comm), settings(_settings), props(_props) {}

  ~bs0_t();

  //setup
  static bs0_t& Setup(occa::device& _device, MPI_Comm& _comm,
        settings_t& _settings, occa::properties& _props);

  void Run();
};


#endif

