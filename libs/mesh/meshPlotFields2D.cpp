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
#include "mesh2D.hpp"

// interpolate data to plot nodes and save to file (one per process
void mesh2D::PlotFields(dfloat* Q, int Nfields, char *fileName){

  FILE *fp;

  fp = fopen(fileName, "w");

  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n");
  fprintf(fp, "  <UnstructuredGrid>\n");
  fprintf(fp, "    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n",
          Nelements*plotNp,
          Nelements*plotNelements);

  // write out nodes
  fprintf(fp, "      <Points>\n");
  fprintf(fp, "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">\n");

  // compute plot node coordinates on the fly
  for(dlong e=0;e<Nelements;++e){
    for(int n=0;n<plotNp;++n){
      dfloat plotxn = 0, plotyn = 0;

      for(int m=0;m<Np;++m){
        plotxn += plotInterp[n*Np+m]*x[m+e*Np];
        plotyn += plotInterp[n*Np+m]*y[m+e*Np];
      }

      fprintf(fp, "       ");
      fprintf(fp, "%g %g %g\n", plotxn,plotyn,0.0);
    }
  }
  fprintf(fp, "        </DataArray>\n");
  fprintf(fp, "      </Points>\n");


  // write out fields
  fprintf(fp, "      <PointData Scalars=\"scalars\">\n");
  fprintf(fp, "        <DataArray type=\"Float32\" Name=\"Fields\" NumberOfComponents=\"%d\" Format=\"ascii\">\n", Nfields);
  for(dlong e=0;e<Nelements;++e){
    for(int n=0;n<plotNp;++n){
      dfloat plotqn[Nfields];

      for (int f=0;f<Nfields;f++) plotqn[f] = 0;

      for(int m=0;m<Np;++m){
        for (int f=0;f<Nfields;f++) {
          dfloat qm = Q[e*Np*Nfields+f*Np+m];
          plotqn[f] += plotInterp[n*Np+m]*qm;
        }
      }

      fprintf(fp, "       ");
      for (int f=0;f<Nfields;f++)
        fprintf(fp, "%g ", plotqn[f]);
      fprintf(fp, "\n");
    }
  }
  fprintf(fp, "       </DataArray>\n");

  fprintf(fp, "     </PointData>\n");

  fprintf(fp, "    <Cells>\n");
  fprintf(fp, "      <DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">\n");

  for(dlong e=0;e<Nelements;++e){
    for(int n=0;n<plotNelements;++n){
      fprintf(fp, "       ");
      for(int m=0;m<plotNverts;++m){
        fprintf(fp, "%d ", e*plotNp + plotEToV[n*plotNverts+m]);
      }
      fprintf(fp, "\n");
    }
  }
  fprintf(fp, "        </DataArray>\n");

  fprintf(fp, "        <DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">\n");
  dlong cnt = 0;
  for(dlong e=0;e<Nelements;++e){
    for(int n=0;n<plotNelements;++n){
      cnt += plotNverts;
      fprintf(fp, "       ");
      fprintf(fp, "%d\n", cnt);
    }
  }
  fprintf(fp, "       </DataArray>\n");

  fprintf(fp, "       <DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">\n");
  for(dlong e=0;e<Nelements;++e){
    for(int n=0;n<plotNelements;++n){
      fprintf(fp, "5\n");
    }
  }
  fprintf(fp, "        </DataArray>\n");
  fprintf(fp, "      </Cells>\n");
  fprintf(fp, "    </Piece>\n");
  fprintf(fp, "  </UnstructuredGrid>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);

}
