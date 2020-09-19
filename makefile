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
	 make clean
	 make clean-libs
	 make realclean
	 make info
	 make help

Usage:

make all
	 Builds all benchmark executables.
make clean
	 Cleans all executables and object files.
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

ifeq (,$(filter all clean clean-libs \
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
BS_CEED_LIBS=mesh ogs linAlg core

.PHONY: all BS ceed_libs \
		clean clean-libs \
		realclean info help

all: BS

ceed_libs:
ifneq (,${verbose})
	${MAKE} -C ${CEED_LIBS_DIR} $(BS_CEED_LIBS) verbose=${verbose}
else
	@${MAKE} -C ${CEED_LIBS_DIR} $(BS_CEED_LIBS) --no-print-directory
endif

BS: ceed_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

#cleanup
clean:
	${MAKE} -C BS clean

clean-libs: clean
	${MAKE} -C ${CEED_LIBS_DIR} clean

realclean: clean
	${MAKE} -C ${CEED_LIBS_DIR} realclean

help:
	$(info $(value CEED_HELP_MSG))
	@true

info:
	$(info OCCA_DIR  = $(OCCA_DIR))
	$(info CEED_DIR  = $(CEED_DIR))
	$(info CEED_ARCH = $(CEED_ARCH))
	$(info CXXFLAGS  = $(CXXFLAGS))
	@true
