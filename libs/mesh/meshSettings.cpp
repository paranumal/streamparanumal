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

#include "mesh.hpp"

void meshAddSettings(settings_t& settings) {

  settings.newSetting("-e", "--elements",
                      "ELEMENT TYPE",
                      "Hex",
                      "Type of mesh elements",
                      {"Tri","Quad","Tet","Hex"});

  settings.newSetting("-nx", "--dimx",
                      "BOX NX",
                      "10",
                      "Number of elements in X-dimension per rank");
  settings.newSetting("-ny", "--dimy",
                      "BOX NY",
                      "10",
                      "Number of elements in Y-dimension per rank");
  settings.newSetting("-nz", "--dimz",
                      "BOX NZ",
                      "10",
                      "Number of elements in Z-dimension per rank");

  settings.newSetting("-p", "--degree",
                      "POLYNOMIAL DEGREE",
                      "4",
                      "Degree of polynomial finite element space",
                      {"1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"});
}

void meshReportSettings(settings_t& settings) {

  settings.reportSetting("ELEMENT TYPE");

  //report the box settings
  settings.reportSetting("BOX NX");
  settings.reportSetting("BOX NY");

  if (settings.compareSetting("ELEMENT TYPE", "Tet") ||
      settings.compareSetting("ELEMENT TYPE", "Hex") )
    settings.reportSetting("BOX NZ");

  settings.reportSetting("POLYNOMIAL DEGREE");
}