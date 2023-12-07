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

//settings for bs solver
bsSettings_t::bsSettings_t(const int argc, char** argv, comm_t _comm):
  settings_t(_comm) {

  platformAddSettings(*this);
  meshAddSettings(*this);

  newSetting("-P", "--problem",
             "BENCHMARK PROBLEM",
             "1",
             "Benchmark problem to run",
             {"0","1","2","3","4","5","6","7","8"});

  newSetting("-b", "--bytes",
             "BYTES",
             "1073741824",
             "Array size in bytes");

  newToggle("-sw", "--sweep",
            "SWEEP",
            "FALSE",
            "Sweep through range of buffer sizes");

  newToggle("-t", "--tuning",
            "KERNEL TUNING",
            "FALSE",
            "Run tuning sweep on kernel");

  newToggle("-v", "--verbose",
             "VERBOSE",
             "FALSE",
             "Enable verbose output");

  newToggle("-ga", "--gpu-aware-mpi",
            "GPU-AWARE MPI",
            "FALSE",
            "Enable direct access of GPU memory in MPI");

  parseSettings(argc, argv);
}

void bsSettings_t::report() {

  std::cout << "Settings:\n\n";
  platformReportSettings(*this);

  int problemNumber=0;
  getSetting("BENCHMARK PROBLEM", problemNumber);

  if (problemNumber==6 ||
      problemNumber==7 ||
      problemNumber==8) {
    meshReportSettings(*this);
  } else if (problemNumber!=0){
    reportSetting("BYTES");
  }
}
