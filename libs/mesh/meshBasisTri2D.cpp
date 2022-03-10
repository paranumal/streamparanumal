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

namespace libp {

// ------------------------------------------------------------------------
// TRI 2D NODES
// ------------------------------------------------------------------------
void mesh_t::NodesTri2D(int _N, dfloat _r[], dfloat _s[]){

  int _Np = (_N+1)*(_N+2)/2;

  EquispacedNodesTri2D(_N, _r, _s); //make equispaced nodes on reference triangle
  WarpBlendTransformTri2D(_N, _Np, _r, _s); //apply warp&blend transform
}

void mesh_t::FaceNodesTri2D(int _N, dfloat _r[], dfloat _s[], int _faceNodes[]){
  int _Nfp = _N+1;
  int _Np = (_N+1)*(_N+2)/2;

  int cnt[3];
  for (int i=0;i<3;i++) cnt[i]=0;

  dfloat deps = 1.;
  while((1.+deps)>1.)
    deps *= 0.5;

  const dfloat NODETOL = 1000.*deps;

  for (int n=0;n<_Np;n++) {
    if(fabs(_s[n]+1)<NODETOL)
      _faceNodes[0*_Nfp+(cnt[0]++)] = n;
    if(fabs(_r[n]+_s[n])<NODETOL)
      _faceNodes[1*_Nfp+(cnt[1]++)] = n;
    if(fabs(_r[n]+1)<NODETOL)
      _faceNodes[2*_Nfp+(cnt[2]++)] = n;
  }
}

void mesh_t::VertexNodesTri2D(int _N, dfloat _r[], dfloat _s[], int _vertexNodes[]){
  int _Np = (_N+1)*(_N+2)/2;

  dfloat deps = 1.;
  while((1.+deps)>1.)
    deps *= 0.5;

  const dfloat NODETOL = 1000.*deps;

  for(int n=0;n<_Np;++n){
    if( (_r[n]+1)*(_r[n]+1)+(_s[n]+1)*(_s[n]+1)<NODETOL)
      _vertexNodes[0] = n;
    if( (_r[n]-1)*(_r[n]-1)+(_s[n]+1)*(_s[n]+1)<NODETOL)
      _vertexNodes[1] = n;
    if( (_r[n]+1)*(_r[n]+1)+(_s[n]-1)*(_s[n]-1)<NODETOL)
      _vertexNodes[2] = n;
  }
}

// Create equidistributed nodes on reference triangle
void mesh_t::EquispacedNodesTri2D(int _N, dfloat _r[], dfloat _s[]){

  int sk = 0;
  for (int n=0;n<_N+1;n++) {
    for (int m=0;m<_N+1-n;m++) {
      _r[sk] = -1.0 + 2.0*m/_N;
      _s[sk] = -1.0 + 2.0*n/_N;
      sk++;
    }
  }
}

/*Find a matching array between nodes on matching faces */
void mesh_t::FaceNodeMatchingTri2D(int _N, dfloat _r[], dfloat _s[],
                                   int _faceNodes[], int R[]){

  int _Nfp = (_N+1);

  const dfloat NODETOL = 1.0e-5;

  dfloat V[2] = {-1.0, 1.0};

  dfloat EX0[Nverts];
  dfloat EX1[Nverts];

  memory<dfloat> x0(_Nfp);
  memory<dfloat> x1(_Nfp);

  for (int fM=0;fM<Nfaces;fM++) {

    for (int v=0;v<Nverts;v++) {
      EX0[v] = 0.0;
    }
    //setup top element with face fM on the bottom
    for (int v=0;v<NfaceVertices;v++) {
      int fv = faceVertices[fM*NfaceVertices + v];
      EX0[fv] = V[v];
    }

    for(int n=0;n<_Nfp;++n){ /* for each face node */
      const int fn = _faceNodes[fM*_Nfp+n];

      /* (r,s) coordinates of interpolation nodes*/
      dfloat rn = _r[fn];
      dfloat sn = _s[fn];

      /* physical coordinate of interpolation node */
      x0[n] = -0.5*(rn+sn)*EX0[0]
             + 0.5*(1+rn)*EX0[1]
             + 0.5*(1+sn)*EX0[2];
    }

    for (int fP=0;fP<Nfaces;fP++) { /*For each neighbor face */
      for (int rot=0;rot<NfaceVertices;rot++) { /* For each face rotation */
        // Zero vertices
        for (int v=0;v<Nverts;v++) {
          EX1[v] = 0.0;
        }
        //setup bottom element with face fP on the top
        for (int v=0;v<NfaceVertices;v++) {
          int fv = faceVertices[fP*NfaceVertices + ((v+rot)%NfaceVertices)];
          EX1[fv] = V[v];
        }

        for(int n=0;n<_Nfp;++n){ /* for each node */
          const int fn = _faceNodes[fP*_Nfp+n];

          /* (r,s,t) coordinates of interpolation nodes*/
          dfloat rn = _r[fn];
          dfloat sn = _s[fn];

          /* physical coordinate of interpolation node */
          x1[n] = -0.5*(rn+sn)*EX1[0]
                 + 0.5*(1+rn)*EX1[1]
                 + 0.5*(1+sn)*EX1[2];
        }

        /* for each node on this face find the neighbor node */
        for(int n=0;n<_Nfp;++n){
          const dfloat xM = x0[n];

          int m=0;
          for(;m<_Nfp;++m){ /* for each neighbor node */
            const dfloat xP = x1[m];

            /* distance between target and neighbor node */
            const dfloat dist = pow(xM-xP,2);

            /* if neighbor node is close to target, match */
            if(dist<NODETOL){
              R[fM*Nfaces*NfaceVertices*_Nfp
                + fP*NfaceVertices*_Nfp
                + rot*_Nfp + n] = m;
              break;
            }
          }

          /*Check*/
          const dfloat xP = x1[m];

          /* distance between target and neighbor node */
          const dfloat dist = pow(xM-xP,2);
          //This shouldn't happen
          LIBP_ABORT("Unable to match face node, face: " << fM
                     << ", matching face: " << fP
                     << ", rotation: " << rot
                     << ", node: " << n
                     << ". Is the reference node set not symmetric?",
                     dist>NODETOL);
        }
      }
    }
  }
}

// ------------------------------------------------------------------------
// Warp & Blend routines
//  Warburton, T. (2006). An explicit construction of interpolation nodes on the simplex.
//                       Journal of engineering mathematics, 56(3), 247-262.
// ------------------------------------------------------------------------

void mesh_t::Warpfactor(int _N, int Npoints, dfloat _r[], dfloat warp[]) {
  // Compute scaled warp function at order N
  // based on rout interpolation nodes

  // Compute GLL and equidistant node distribution
  memory<dfloat> GLLr(_N+1);
  memory<dfloat> req (_N+1);
  JacobiGLL(_N, GLLr.ptr());
  EquispacedNodes1D(_N, req.ptr());

  // Make interpolation from req to r
  memory<dfloat> I((_N+1)*Npoints);
  InterpolationMatrix1D(_N, _N+1, req.ptr(), Npoints, _r, I.ptr());

  // Compute warp factor
  for (int n=0;n<Npoints;n++) {
    warp[n] = 0.0;

    for (int i=0;i<_N+1;i++) {
      warp[n] += I[n*(_N+1)+i]*(GLLr[i] - req[i]);
    }

    // Scale factor
    dfloat zerof = (std::abs(_r[n])<1.0-1.0e-10) ? 1 : 0;
    dfloat sf = 1.0 - (zerof*_r[n])*(zerof*_r[n]);
    warp[n] = warp[n]/sf + warp[n]*(zerof-1);
  }
}

static void xytors(int Npoints, dfloat x[], dfloat y[], dfloat r[], dfloat s[]) {
  for (int n=0;n<Npoints;n++) {
    dfloat L1 = (sqrt(3.0)*y[n]+1.0)/3.0;
    dfloat L2 = (-3.0*x[n] - sqrt(3.0)*y[n] + 2.0)/6.0;
    dfloat L3 = ( 3.0*x[n] - sqrt(3.0)*y[n] + 2.0)/6.0;

    r[n] = -L2 + L3 - L1; s[n] = -L2 - L3 + L1;
  }
}

void mesh_t::WarpBlendTransformTri2D(int _N, int _Npoints, dfloat _r[], dfloat _s[], dfloat alphaIn){

  const dfloat alpopt[15] = {0.0000, 0.0000, 1.4152, 0.1001, 0.2751, 0.9800, 1.0999,
                             1.2832, 1.3648, 1.4773, 1.4959, 1.5743, 1.5770, 1.6223, 1.6258};

  dfloat alpha;
  if (alphaIn==-1) {
    if (_N<16) {
      alpha = alpopt[_N-1];
    } else {
      alpha = 5./3.;
    }
  } else {
    alpha = alphaIn;
  }

  // Convert r s coordinates to points in equilateral triangle
  memory<dfloat> L1(_Npoints);
  memory<dfloat> L2(_Npoints);
  memory<dfloat> L3(_Npoints);

  memory<dfloat> dL32(_Npoints);
  memory<dfloat> dL13(_Npoints);
  memory<dfloat> dL21(_Npoints);

  memory<dfloat> _x(_Npoints);
  memory<dfloat> _y(_Npoints);

  for (int n=0;n<_Npoints;n++) {
    L1[n] =  0.5*(1.+_s[n]);
    L2[n] = -0.5*(_r[n]+_s[n]);
    L3[n] =  0.5*(1.+_r[n]);

    dL32[n] = L3[n]-L2[n];
    dL13[n] = L1[n]-L3[n];
    dL21[n] = L2[n]-L1[n];

    _x[n] = -L2[n]+L3[n]; _y[n] = (-L2[n]-L3[n]+2.*L1[n])/sqrt(3.0);
  }

  memory<dfloat> warpf1(_Npoints);
  memory<dfloat> warpf2(_Npoints);
  memory<dfloat> warpf3(_Npoints);

  Warpfactor(_N, _Npoints, dL32.ptr(), warpf1.ptr());
  Warpfactor(_N, _Npoints, dL13.ptr(), warpf2.ptr());
  Warpfactor(_N, _Npoints, dL21.ptr(), warpf3.ptr());

  for (int n=0;n<_Npoints;n++) {
    dfloat blend1 = 4.0*L2[n]*L3[n];
    dfloat blend2 = 4.0*L3[n]*L1[n];
    dfloat blend3 = 4.0*L1[n]*L2[n];

    dfloat warp1 = blend1*warpf1[n]*(1.0+alpha*alpha*L1[n]*L1[n]);
    dfloat warp2 = blend2*warpf2[n]*(1.0+alpha*alpha*L2[n]*L2[n]);
    dfloat warp3 = blend3*warpf3[n]*(1.0+alpha*alpha*L3[n]*L3[n]);

    _x[n] += 1.*warp1 + cos(2.*M_PI/3.)*warp2 + cos(4.*M_PI/3.)*warp3;
    _y[n] += 0.*warp1 + sin(2.*M_PI/3.)*warp2 + sin(4.*M_PI/3.)*warp3;
  }

  xytors(_Npoints, _x.ptr(), _y.ptr(), _r, _s);
}

} //namespace libp
