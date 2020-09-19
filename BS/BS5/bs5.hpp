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

#ifndef BS5_HPP
#define BS5_HPP 1

#include "platform.hpp"

#define DBS5 CEED_DIR"/BS/BS5/"

class bs5Settings_t: public settings_t {
public:
  bs5Settings_t(const int argc, char** argv, MPI_Comm& _comm);
  void report();
};

class bs5_t {
public:
  platform_t &platform;
  settings_t &settings;

  int blockSize;
  int Nv;

  occa::kernel kernel1, kernel2;

  bs5_t() = delete;
  bs5_t(platform_t &_platform, settings_t& _settings):
    platform(_platform), settings(_settings) {}

  ~bs5_t();

  //setup
  static bs5_t& Setup(platform_t &_platform, settings_t& _settings);

  void Run();
};


#endif

