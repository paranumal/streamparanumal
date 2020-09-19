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

void hipReadTemperatures(int dev,double *T, double *freq){
  
  // should make this robust
  char fname[BUFSIZ];

  // read temps
  for(int probe=1;probe<=3;++probe){
    sprintf(fname, "/sys/class/hwmon/hwmon%d/temp%d_input", dev, probe);
    
    FILE *fp = fopen(fname, "r");
    T[probe-1] = 0;
    
    if(fp){
      int val;
      fscanf(fp, "%d", &val);
      fclose(fp);
      T[probe-1] = val/1000.;
    }
  }
  
  sprintf(fname, "/sys/class/hwmon/hwmon%d/freq1_input", dev);
  
  FILE *fp = fopen(fname, "r");
  *freq = 0;
  
  if(fp){
    int val;
    fscanf(fp, "%d", &val);
    fclose(fp);
    *freq = val/1.e9;
  }
  
}
