# GSLIB 

[![Build Status](https://travis-ci.org/gslib/gslib.svg?branch=master)](https://travis-ci.org/gslib/gslib)

* Scalable Many-to-Many collectives
* Robust interpolation for hexahedral spectral element meshes

# Build Instructions

The build system relies on GNU Make with the `make` command. To compile gslib just run:

```
make CC=mpicc FC=mpif77
make PREFIX=<install path> install
```

# Applications

**\[1]&#160;[Nek5000](https://nek5000.mcs.anl.gov/)**: Nek5000 open-source, spectral element code.

**\[2]&#160;[CEED](http://ceed.exascaleproject.org/)**: Co-design center for Efficient Exascale Discretizations.

**\[3]&#160;[Nektar++](http://www.nektar.info)**: Nektar++ open-source spectral/hp element code.
