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

CEED Bakeoff Streaming makefile targets:

	 make BS (default)
	 make BS{#}
	 make clean
	 make clean-libs
	 make realclean
	 make info
	 make help

Usage:

make BS
	 Builds each Bakeoff Kernel executable.
make BS{#}
	 Builds a Bakeoff Kernel executable,
	 # can be 1-8.
make clean
	 Cleans all BS executables and object files.
make clean-BS{#}
	 Cleans a BS executable and object files.
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

ifeq (,$(filter BS \
								BS1 BS2 BS3 BS4 BS5 BS6 \
								clean-BS0 clean-BS1 clean-BS2 clean-BS3 clean-BS4 clean-BS5 clean-BS6 \
								clean clean-libs \
								realclean info help,$(MAKECMDGOALS)))
ifneq (,$(MAKECMDGOALS))
$(error ${CEED_HELP_MSG})
endif
endif

ifndef CEED_MAKETOP_LOADED
ifeq (,$(wildcard ../make.top))
$(error cannot locate ${PWD}/../make.top)
else
include ../make.top
endif
endif

#libraries
GS_DIR       =${CEED_TPL_DIR}/gslib
BLAS_DIR     =${CEED_TPL_DIR}/BlasLapack
CORE_DIR     =${CEED_DIR}/core
OGS_DIR      =${CORE_DIR}/ogs
MESH_DIR     =${CORE_DIR}/mesh

.PHONY: all BS \
				BS1 BS2 BS3 BS4 BS5 BS6 BS7 BS8 \
				clean-BS1 clean-BS2 clean-BS3 clean-BS4 clean-BS5 \
				clean-BS6 clean-BS7 clean-BS8 \
				clean clean-libs \
				realclean info help

all: BS

BS: BS1 BS2 BS3 BS4 BS5 BS6 BS7 BS8

BS1: libmesh
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS2: libmesh
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS3: libmesh
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS4: libmesh
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS5: libmesh
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS6: libmesh
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS7: libmesh
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS8: libmesh
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
clean: clean-BS1 clean-BS2 clean-BS3 clean-BS4 clean-BS5 clean-BS6

clean-BS1:
	${MAKE} -C BS1 clean

clean-BS2:
	${MAKE} -C BS2 clean

clean-BS3:
	${MAKE} -C BS3 clean

clean-BS4:
	${MAKE} -C BS4 clean

clean-BS5:
	${MAKE} -C BS5 clean

clean-BS6:
	${MAKE} -C BS6 clean

clean-BS7:
	${MAKE} -C BS6 clean

clean-BS8:
	${MAKE} -C BS6 clean

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
