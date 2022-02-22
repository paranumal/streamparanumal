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
// TET 3D NODES
// ------------------------------------------------------------------------
void mesh_t::NodesTet3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[]){

  int _Np = (_N+1)*(_N+2)*(_N+3)/6;

  EquispacedNodesTet3D(_N, _r, _s, _t); //make equispaced nodes on reference tet
  WarpBlendTransformTet3D(_N, _Np, _r, _s, _t); //apply warp&blend transform
}

void mesh_t::FaceNodesTet3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[], int _faceNodes[]){
  int _Nfp = (_N+1)*(_N+2)/2;
  int _Np = (_N+1)*(_N+2)*(_N+3)/6;

  int cnt[4];
  for (int i=0;i<4;i++) cnt[i]=0;

  dfloat deps = 1.;
  while((1.+deps)>1.)
    deps *= 0.5;

  const dfloat NODETOL = 1000.*deps;

  for (int n=0;n<_Np;n++) {
    if(fabs(_t[n]+1)<NODETOL)
      _faceNodes[0*_Nfp+(cnt[0]++)] = n;
    if(fabs(_s[n]+1)<NODETOL)
      _faceNodes[1*_Nfp+(cnt[1]++)] = n;
    if(fabs(_r[n]+_s[n]+_t[n]+1.0)<NODETOL)
      _faceNodes[2*_Nfp+(cnt[2]++)] = n;
    if(fabs(_r[n]+1)<NODETOL)
      _faceNodes[3*_Nfp+(cnt[3]++)] = n;
  }
}

void mesh_t::VertexNodesTet3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[], int _vertexNodes[]){
  int _Np = (_N+1)*(_N+2)*(_N+3)/6;

  dfloat deps = 1.;
  while((1.+deps)>1.)
    deps *= 0.5;

  const dfloat NODETOL = 1000.*deps;

  for(int n=0;n<_Np;++n){
    if( (_r[n]+1)*(_r[n]+1)+(_s[n]+1)*(_s[n]+1)+(_t[n]+1)*(_t[n]+1)<NODETOL)
      _vertexNodes[0] = n;
    if( (_r[n]-1)*(_r[n]-1)+(_s[n]+1)*(_s[n]+1)+(_t[n]+1)*(_t[n]+1)<NODETOL)
      _vertexNodes[1] = n;
    if( (_r[n]+1)*(_r[n]+1)+(_s[n]-1)*(_s[n]-1)+(_t[n]+1)*(_t[n]+1)<NODETOL)
      _vertexNodes[2] = n;
    if( (_r[n]+1)*(_r[n]+1)+(_s[n]+1)*(_s[n]+1)+(_t[n]-1)*(_t[n]-1)<NODETOL)
      _vertexNodes[3] = n;
  }
}

// Create equidistributed nodes on reference tet
void mesh_t::EquispacedNodesTet3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[]){

  int sk = 0;
  for (int k=0;k<_N+1;k++) {
    for (int n=0;n<_N+1-k;n++) {
      for (int m=0;m<_N+1-n-k;m++) {
        _r[sk] = -1.0 + 2.0*m/_N;
        _s[sk] = -1.0 + 2.0*n/_N;
        _t[sk] = -1.0 + 2.0*k/_N;
        sk++;
      }
    }
  }
}

/*Find a matching array between nodes on matching faces */
void mesh_t::FaceNodeMatchingTet3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[],
                                   int _faceNodes[], int R[]){

  int _Nfp = (_N+1)*(_N+2)/2;

  const dfloat NODETOL = 1.0e-5;

  dfloat V0[3][2] = {{-1.0,-1.0},{ 1.0,-1.0},{-1.0, 1.0}};
  dfloat V1[3][2] = {{-1.0,-1.0},{-1.0, 1.0},{ 1.0,-1.0}};

  dfloat EX0[Nverts], EY0[Nverts];
  dfloat EX1[Nverts], EY1[Nverts];

  libp::memory<dfloat> x0(_Nfp);
  libp::memory<dfloat> y0(_Nfp);

  libp::memory<dfloat> x1(_Nfp);
  libp::memory<dfloat> y1(_Nfp);


  for (int fM=0;fM<Nfaces;fM++) {

    for (int v=0;v<Nverts;v++) {
      EX0[v] = 0.0; EY0[v] = 0.0;
    }
    //setup top element with face fM on the bottom
    for (int v=0;v<NfaceVertices;v++) {
      int fv = faceVertices[fM*NfaceVertices + v];
      EX0[fv] = V0[v][0]; EY0[fv] = V0[v][1];
    }

    for(int n=0;n<_Nfp;++n){ /* for each face node */
      const int fn = _faceNodes[fM*_Nfp+n];

      /* (r,s,t) coordinates of interpolation nodes*/
      dfloat rn = _r[fn];
      dfloat sn = _s[fn];
      dfloat tn = _t[fn];

      /* physical coordinate of interpolation node */
      x0[n] = -0.5*(1+rn+sn+tn)*EX0[0]
             + 0.5*(1+rn)*EX0[1]
             + 0.5*(1+sn)*EX0[2]
             + 0.5*(1+tn)*EX0[3];
      y0[n] = -0.5*(1+rn+sn+tn)*EY0[0]
             + 0.5*(1+rn)*EY0[1]
             + 0.5*(1+sn)*EY0[2]
             + 0.5*(1+tn)*EY0[3];
    }

    for (int fP=0;fP<Nfaces;fP++) { /*For each neighbor face */
      for (int rot=0;rot<NfaceVertices;rot++) { /* For each face rotation */
        // Zero vertices
        for (int v=0;v<Nverts;v++) {
          EX1[v] = 0.0; EY1[v] = 0.0;
        }
        //setup bottom element with face fP on the top
        for (int v=0;v<NfaceVertices;v++) {
          int fv = faceVertices[fP*NfaceVertices + ((v+rot)%NfaceVertices)];
          EX1[fv] = V1[v][0]; EY1[fv] = V1[v][1];
        }

        for(int n=0;n<_Nfp;++n){ /* for each node */
          const int fn = _faceNodes[fP*_Nfp+n];

          /* (r,s,t) coordinates of interpolation nodes*/
          dfloat rn = _r[fn];
          dfloat sn = _s[fn];
          dfloat tn = _t[fn];

          /* physical coordinate of interpolation node */
          x1[n] = -0.5*(1+rn+sn+tn)*EX1[0]
                 + 0.5*(1+rn)*EX1[1]
                 + 0.5*(1+sn)*EX1[2]
                 + 0.5*(1+tn)*EX1[3];
          y1[n] = -0.5*(1+rn+sn+tn)*EY1[0]
                 + 0.5*(1+rn)*EY1[1]
                 + 0.5*(1+sn)*EY1[2]
                 + 0.5*(1+tn)*EY1[3];
        }

        /* for each node on this face find the neighbor node */
        for(int n=0;n<_Nfp;++n){
          const dfloat xM = x0[n];
          const dfloat yM = y0[n];

          int m=0;
          for(;m<_Nfp;++m){ /* for each neighbor node */
            const dfloat xP = x1[m];
            const dfloat yP = y1[m];

            /* distance between target and neighbor node */
            const dfloat dist = pow(xM-xP,2) + pow(yM-yP,2);

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
          const dfloat yP = y1[m];

          /* distance between target and neighbor node */
          const dfloat dist = pow(xM-xP,2) + pow(yM-yP,2);
          if(dist>NODETOL){
            //This shouldn't happen
            std::stringstream ss;
            ss << "Unable to match face node, face: " << fM
               << ", matching face: " << fP
               << ", rotation: " << rot
               << ", node: " << n
               << ". Is the reference node set not symmetric?";
            LIBP_ABORT(ss.str())
          }
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

static void xyztorst(int Npoints, dfloat x[], dfloat y[], dfloat z[], dfloat r[], dfloat s[], dfloat t[]) {
  // vertices of tetrahedron
  dfloat v1[3] = {-1.0, -1./sqrt(3.), -1./sqrt(6.)};
  dfloat v2[3] = { 1.0, -1./sqrt(3.), -1./sqrt(6.)};
  dfloat v3[3] = { 0.0,  2./sqrt(3.), -1./sqrt(6.)};
  dfloat v4[3] = { 0.0,  0.,           3./sqrt(6.)};

  libp::memory<dfloat> XYZ(3*Npoints);
  libp::memory<dfloat> RST(3*Npoints);
  libp::memory<dfloat> A(3*3);

  for (int i=0;i<3;i++) {
    A[0*3+i] = 0.5*(v2[i]-v1[i]);
    A[1*3+i] = 0.5*(v3[i]-v1[i]);
    A[2*3+i] = 0.5*(v4[i]-v1[i]);
  }

  for (int n=0;n<Npoints;n++) {
    XYZ[3*n+0] = x[n]-0.5*(v2[0]+v3[0]+v4[0]-v1[0]);
    XYZ[3*n+1] = y[n]-0.5*(v2[1]+v3[1]+v4[1]-v1[1]);
    XYZ[3*n+2] = z[n]-0.5*(v2[2]+v3[2]+v4[2]-v1[2]);
  }

  matrixRightSolve(Npoints, 3, XYZ.ptr(), 3, 3, A.ptr(), RST.ptr());

  for (int n=0;n<Npoints;n++) {
    r[n] = RST[3*n+0];
    s[n] = RST[3*n+1];
    t[n] = RST[3*n+2];
  }
}

void mesh_t::WarpShiftFace3D(int _N, int Npoints, dfloat alpha,
                             dfloat L1[], dfloat L2[], dfloat L3[],
                             dfloat w1[], dfloat w2[]) {
  // Compute scaled warp function at order N
  // based on rout interpolation nodes

  libp::memory<dfloat> dL32(Npoints);
  libp::memory<dfloat> dL13(Npoints);
  libp::memory<dfloat> dL21(Npoints);

  libp::memory<dfloat> warpf1(Npoints);
  libp::memory<dfloat> warpf2(Npoints);
  libp::memory<dfloat> warpf3(Npoints);

  for (int n=0;n<Npoints;n++) {
    dL32[n] = L3[n]-L2[n];
    dL13[n] = L1[n]-L3[n];
    dL21[n] = L2[n]-L1[n];
  }

  Warpfactor(_N, Npoints, dL32.ptr(), warpf1.ptr());
  Warpfactor(_N, Npoints, dL13.ptr(), warpf2.ptr());
  Warpfactor(_N, Npoints, dL21.ptr(), warpf3.ptr());

  for (int n=0;n<Npoints;n++) {
    dfloat blend1 = 4.0*L2[n]*L3[n];
    dfloat blend2 = 4.0*L3[n]*L1[n];
    dfloat blend3 = 4.0*L1[n]*L2[n];

    dfloat warp1 = blend1*warpf1[n]*(1.0+alpha*alpha*L1[n]*L1[n]);
    dfloat warp2 = blend2*warpf2[n]*(1.0+alpha*alpha*L2[n]*L2[n]);
    dfloat warp3 = blend3*warpf3[n]*(1.0+alpha*alpha*L3[n]*L3[n]);

    w1[n] = 1.*warp1 + cos(2.*M_PI/3.)*warp2 + cos(4.*M_PI/3.)*warp3;
    w2[n] = 0.*warp1 + sin(2.*M_PI/3.)*warp2 + sin(4.*M_PI/3.)*warp3;
  }
}

void mesh_t::WarpBlendTransformTet3D(int _N, int _Npoints, dfloat _r[], dfloat _s[], dfloat _t[], dfloat alphaIn){

  const dfloat alpopt[15] = {0.0000,0.0000,0.00000,0.1002,1.1332,1.5608,1.3413,
                             1.2577,1.1603,1.10153,0.6080,0.4523,0.8856,0.8717,0.9655};

  dfloat alpha;
  if (alphaIn==-1) {
    if (_N<16) {
      alpha = alpopt[_N-1];
    } else {
      alpha = 1.;
    }
  } else {
    alpha = alphaIn;
  }

  // vertices of tetrahedron
  dfloat v1[3] = {-1.0, -1./sqrt(3.), -1./sqrt(6.)};
  dfloat v2[3] = { 1.0, -1./sqrt(3.), -1./sqrt(6.)};
  dfloat v3[3] = { 0.0,  2./sqrt(3.), -1./sqrt(6.)};
  dfloat v4[3] = { 0.0,  0.,           3./sqrt(6.)};

  // orthogonal axis tangents on faces 1-4
  dfloat t1[4][4], t2[4][4];
  for (int v=0;v<3;v++) {
    t1[0][v] = v2[v]-v1[v];             t1[1][v] = v2[v]-v1[v];
    t1[2][v] = v3[v]-v2[v];             t1[3][v] = v3[v]-v1[v];
    t2[0][v] = v3[v]-0.5*(v1[v]+v2[v]); t2[1][v] = v4[v]-0.5*(v1[v]+v2[v]);
    t2[2][v] = v4[v]-0.5*(v2[v]+v3[v]); t2[3][v] = v4[v]-0.5*(v1[v]+v3[v]);
  }
  // normalize tangents
  for (int v=0;v<4;v++) {
    dfloat normt1 = sqrt(t1[v][0]*t1[v][0]+t1[v][1]*t1[v][1]+t1[v][2]*t1[v][2]);
    dfloat normt2 = sqrt(t2[v][0]*t2[v][0]+t2[v][1]*t2[v][1]+t2[v][2]*t2[v][2]);
    for (int i=0;i<3;i++) {
      t1[v][i] /= normt1;
      t2[v][i] /= normt2;
    }
  }

  // Convert r s coordinates to points in equilateral triangle
  libp::memory<dfloat> L1(_Npoints);
  libp::memory<dfloat> L2(_Npoints);
  libp::memory<dfloat> L3(_Npoints);
  libp::memory<dfloat> L4(_Npoints);

  libp::memory<dfloat> _x(_Npoints);
  libp::memory<dfloat> _y(_Npoints);
  libp::memory<dfloat> _z(_Npoints);

  libp::memory<dfloat> shiftx(_Npoints, 0.0);
  libp::memory<dfloat> shifty(_Npoints, 0.0);
  libp::memory<dfloat> shiftz(_Npoints, 0.0);

  for (int n=0;n<_Npoints;n++) {
    L1[n] =  0.5*(1.+_t[n]);
    L2[n] =  0.5*(1.+_s[n]);
    L3[n] = -0.5*(1.0+_r[n]+_s[n]+_t[n]);
    L4[n] =  0.5*(1.+_r[n]);

    _x[n] =  L3[n]*v1[0]+L4[n]*v2[0]+L2[n]*v3[0]+L1[n]*v4[0];
    _y[n] =  L3[n]*v1[1]+L4[n]*v2[1]+L2[n]*v3[1]+L1[n]*v4[1];
    _z[n] =  L3[n]*v1[2]+L4[n]*v2[2]+L2[n]*v3[2]+L1[n]*v4[2];
  }

  libp::memory<dfloat> warp1(_Npoints, 0.0);
  libp::memory<dfloat> warp2(_Npoints, 0.0);

  for (int f=0;f<4;f++) {
    libp::memory<dfloat> La, Lb, Lc, Ld;
    if(f==0) {La = L1; Lb = L2; Lc = L3; Ld = L4;}
    if(f==1) {La = L2; Lb = L1; Lc = L3; Ld = L4;}
    if(f==2) {La = L3; Lb = L1; Lc = L4; Ld = L2;}
    if(f==3) {La = L4; Lb = L1; Lc = L3; Ld = L2;}

    // compute warp tangential to face
    WarpShiftFace3D(_N, _Npoints, alpha,
                    Lb.ptr(), Lc.ptr(), Ld.ptr(),
                    warp1.ptr(), warp2.ptr());

    for (int n=0;n<_Npoints;n++) {
      dfloat blend = Lb[n]*Lc[n]*Ld[n];

      // modify linear blend
      dfloat denom = (Lb[n]+.5*La[n])*(Lc[n]+.5*La[n])*(Ld[n]+.5*La[n]);
      if (denom>1.e-10) {
        blend = (1+(alpha*La[n])*(alpha*La[n]))*blend/denom;
      }

      // compute warp & blend
      shiftx[n] += (blend*warp1[n])*t1[f][0] + (blend*warp2[n])*t2[f][0];
      shifty[n] += (blend*warp1[n])*t1[f][1] + (blend*warp2[n])*t2[f][1];
      shiftz[n] += (blend*warp1[n])*t1[f][2] + (blend*warp2[n])*t2[f][2];

      // fix face warp
      if ((La[n]<1.e-10) && ((Lb[n]<1.e-10)||(Lc[n]<1.e-10)||(Ld[n]<1.e-10))) {
        shiftx[n] = warp1[n]*t1[f][0] + warp2[n]*t2[f][0];
        shifty[n] = warp1[n]*t1[f][1] + warp2[n]*t2[f][1];
        shiftz[n] = warp1[n]*t1[f][2] + warp2[n]*t2[f][2];
      }
    }
  }
  for (int n=0;n<_Npoints;n++) {
    _x[n] += shiftx[n];
    _y[n] += shifty[n];
    _z[n] += shiftz[n];
  }

  xyztorst(_Npoints,
           _x.ptr(), _y.ptr(), _z.ptr(),
           _r, _s, _t);
}

} //namespace libp
