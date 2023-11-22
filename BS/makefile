#####################################################################################
#
#The MIT License (MIT)
#
#Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus
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

define STREAM_HELP_MSG

streamParanumal Benchmarks makefile targets:

	 make BS (default)
	 make BS{#}
	 make clean
	 make clean-libs
	 make clean-kernels
	 make realclean
	 make info
	 make help

Usage:

make BS
	 Builds each Bakeoff Kernel executable.
make BS{#}
	 Builds a Bakeoff Kernel executable,
	 # can be 0-8.
make clean
	 Cleans all BS executables and object files.
make clean-BS{#}
	 Cleans a BS executable and object files.
make clean-libs
	 In addition to "make clean", also clean the core, mesh, and ogs libraries.
make clean-kernels
   In addition to "make clean-libs", also cleans the cached OCCA kernels.
make realclean
	 In addition to "make clean-kernels", also clean 3rd party libraries.
make info
	 List directories and compiler flags in use.
make help
	 Display this help message.

Can use "make verbose=true" for verbose output.

endef

ifeq (,$(filter BS \
								BS0 BS1 BS2 BS3 BS4 BS5 BS6 \
								clean-BS0 clean-BS1 clean-BS2 clean-BS3 clean-BS4 clean-BS5 clean-BS6 \
								clean clean-libs clean-kernels \
								realclean info help,$(MAKECMDGOALS)))
ifneq (,$(MAKECMDGOALS))
$(error ${STREAM_HELP_MSG})
endif
endif

ifndef LIBP_MAKETOP_LOADED
ifeq (,$(wildcard ../make.top))
$(error cannot locate ${PWD}/../make.top)
else
include ../make.top
endif
endif

#libraries
STREAM_LIBP_LIBS=mesh ogs core prim

.PHONY: all BS libp_libs \
				BS0 BS1 BS2 BS3 BS4 BS5 BS6 BS7 BS8 \
				clean-BS0 clean-BS1 clean-BS2 clean-BS3 clean-BS4 \
				clean-BS5 clean-BS6 clean-BS7 clean-BS8 \
				clean clean-libs clean-kernels \
				realclean info help

all: BS

BS: BS0 BS1 BS2 BS3 BS4 BS5 BS6 BS7 BS8

${OCCA_DIR}/lib/libocca.so:
	${MAKE} -C ${OCCA_DIR}

libp_libs: ${OCCA_DIR}/lib/libocca.so
ifneq (,${verbose})
	${MAKE} -C ${LIBP_LIBS_DIR} $(STREAM_LIBP_LIBS) verbose=${verbose}
else
	@${MAKE} -C ${LIBP_LIBS_DIR} $(STREAM_LIBP_LIBS) --no-print-directory
endif

BS0: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS1: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS2: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS3: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS4: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS5: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS6: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS7: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

BS8: libp_libs
ifneq (,${verbose})
	${MAKE} -C $(@F) verbose=${verbose}
else
	@printf "%b" "$(SOL_COLOR)Building $(@F) benchmark $(NO_COLOR)\n";
	@${MAKE} -C $(@F) --no-print-directory
endif

#cleanup
clean: clean-BS0 clean-BS1 clean-BS2 clean-BS3 clean-BS4 \
       clean-BS5 clean-BS6 clean-BS7 clean-BS8

clean-BS0:
	${MAKE} -C BS0 clean

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
	${MAKE} -C BS7 clean

clean-BS8:
	${MAKE} -C BS8 clean

clean-libs: clean
	${MAKE} -C ${LIBP_LIBS_DIR} clean

clean-kernels: clean-libs
	rm -rf ${STREAM_DIR}/.occa/

realclean: clean-kernels
	${MAKE} -C ${OCCA_DIR} clean

help:
	$(info $(value STREAM_HELP_MSG))
	@true

info:
	$(info STREAM_DIR  = $(STREAM_DIR))
	$(info OCCA_DIR  = $(OCCA_DIR))
	$(info LIBP_ARCH = $(LIBP_ARCH))
	$(info LIBP_CXXFLAGS  = $(LIBP_CXXFLAGS))
	@true
