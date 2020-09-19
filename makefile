#####################################################################################
#
#The MIT License (MIT)
#
#Copyright (c) 2020 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#
#####################################################################################

define CEED_HELP_MSG

CEED Benchmarks makefile targets:

	 make all (default)
	 make BP/BK/BS
	 make clean
	 make clean-libs
	 make realclean
	 make info
	 make help

Usage:

make all
	 Builds all benchmark executables.
make BP/BK/BS
	 Builds Bakeoff Problems, Kernels, or Streaming executables,
make clean
	 Cleans all executables and object files.
make clean-BP/BK/BS
	 Cleans BP, BK, or BS executables and object files.
make clean-libs
	 In addition to "make clean", also clean the core, mesh, and ogs libraries.
make realclean
	 In addition to "make clean-libs", also clean 3rd party libraries.
make info
	 List directories and compiler flags in use.
make help
	 Display this help message.

Can use "make verbose=true" for verbose output.

endef

ifeq (,$(filter BP BK BS \
				clean-BP clean-BK clean-BS \
				clean clean-libs \
				realclean info help,$(MAKECMDGOALS)))
ifneq (,$(MAKECMDGOALS))
$(error ${CEED_HELP_MSG})
endif
endif

ifndef CEED_MAKETOP_LOADED
ifeq (,$(wildcard make.top))
$(error cannot locate ${PWD}/make.top)
else
include make.top
endif
endif

#libraries
GS_DIR       =${CEED_TPL_DIR}/gslib
BLAS_DIR     =${CEED_TPL_DIR}/BlasLapack
CORE_DIR     =${CEED_DIR}/core
OGS_DIR      =${CORE_DIR}/ogs
MESH_DIR     =${CORE_DIR}/mesh

.PHONY: all BP BK BS \
		clean-BP clean-BK clean-BS \
		clean clean-libs \
		realclean info help

all: BP BK BS

BP: libmesh
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BK: libmesh
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS: libmesh
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

libmesh: libogs libgs libblas libcore
ifneq (,${verbose})
	${MAKE} -C ${MESH_DIR} lib verbose=${verbose}
else
	@${MAKE} -C ${MESH_DIR} lib --no-print-directory
endif

libogs: libcore
ifneq (,${verbose})
	${MAKE} -C ${OGS_DIR} lib verbose=${verbose}
else
	@${MAKE} -C ${OGS_DIR} lib --no-print-directory
endif

libcore: libgs
ifneq (,${verbose})
	${MAKE} -C ${CORE_DIR} lib verbose=${verbose}
else
	@${MAKE} -C ${CORE_DIR} lib --no-print-directory
endif

libgs: libblas
ifneq (,${verbose})
	${MAKE} -C $(GS_DIR) install verbose=${verbose}
else
	@${MAKE} -C $(GS_DIR) install --no-print-directory
endif

libblas:
ifneq (,${verbose})
	${MAKE} -C ${BLAS_DIR} lib verbose=${verbose}
else
	@${MAKE} -C ${BLAS_DIR} lib --no-print-directory
endif

#cleanup
clean: clean-BP clean-BK clean-BS

clean-BP:
	${MAKE} -C BP clean

clean-BK:
	${MAKE} -C BK clean

clean-BS:
	${MAKE} -C BS clean

clean-libs: clean
	${MAKE} -C ${MESH_DIR} clean
	${MAKE} -C ${OGS_DIR} clean
	${MAKE} -C ${CORE_DIR} clean

realclean: clean-libs
	${MAKE} -C ${GS_DIR} clean
	${MAKE} -C ${BLAS_DIR} clean

help:
	$(info $(value CEED_HELP_MSG))
	@true

info:
	$(info OCCA_DIR  = $(OCCA_DIR))
	$(info CEED_DIR  = $(CEED_DIR))
	$(info CEED_ARCH = $(CEED_ARCH))
	$(info CXXFLAGS  = $(CXXFLAGS))
	@true
