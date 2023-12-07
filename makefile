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

define LIBP_HELP_MSG

streamParanumal Benchmarks makefile targets:

	 make all (default)
	 make clean
	 make clean-libs
	 make clean-kernels
	 make realclean
	 make info
	 make help

Usage:

make all
	 Builds the benchmark.
make clean
	 Cleans all executables and object files.
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

ifeq (,$(filter all clean clean-libs clean-kernels \
				realclean info help,$(MAKECMDGOALS)))
ifneq (,$(MAKECMDGOALS))
$(error ${LIBP_HELP_MSG})
endif
endif

ifndef LIBP_MAKETOP_LOADED
ifeq (,$(wildcard make.top))
$(error cannot locate ${PWD}/make.top)
else
include make.top
endif
endif

#libraries
CORE_LIBS=mesh ogs prim core

#includes
INCLUDES=${LIBP_INCLUDES} \
         -I.

#defines
DEFINES =${LIBP_DEFINES} \
         -DLIBP_DIR='"${LIBP_DIR}"'

#.cpp compilation flags
BS_CXXFLAGS=${LIBP_CXXFLAGS} ${DEFINES} ${INCLUDES}

#link libraries
LIBS=-L${LIBP_LIBS_DIR} $(addprefix -l,$(CORE_LIBS)) \
     ${LIBP_LIBS}

#link flags
LFLAGS=${BS_CXXFLAGS} ${LIBS}

#object dependancies
DEPS=$(wildcard src/*.hpp) \
     $(wildcard $(LIBP_INCLUDE_DIR)/*.h) \
     $(wildcard $(LIBP_INCLUDE_DIR)/*.hpp)

SRC =$(wildcard src/*.cpp)

OBJS=$(SRC:.cpp=.o)

.PHONY: all clean clean-libs clean-kernels \
		realclean info help

all: BS

${OCCA_DIR}/lib/libocca.so:
	${MAKE} -C ${OCCA_DIR}

libp_libs: ${OCCA_DIR}/lib/libocca.so
ifneq (,${verbose})
	${MAKE} -C ${LIBP_LIBS_DIR} $(CORE_LIBS) verbose=${verbose}
else
	@${MAKE} -C ${LIBP_LIBS_DIR} $(CORE_LIBS) --no-print-directory
endif

BS:$(OBJS) libp_libs | libp_libs
ifneq (,${verbose})
	$(LIBP_LD) -o BS $(OBJS) $(LFLAGS)
else
	@printf "%b" "$(EXE_COLOR)Linking $(@F)$(NO_COLOR)\n";
	@$(LIBP_LD) -o BS $(OBJS) $(LFLAGS)
endif

# rule for .cpp files
%.o: %.cpp $(DEPS) | libp_libs
ifneq (,${verbose})
	$(LIBP_CXX) -o $*.o -c $*.cpp $(BS_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(LIBP_CXX) -o $*.o -c $*.cpp $(BS_CXXFLAGS)
endif

#cleanup
clean:
	rm -f src/*.o BS

clean-libs: clean
	${MAKE} -C ${LIBP_LIBS_DIR} clean

clean-kernels: clean-libs
	rm -rf ${LIBP_DIR}/.occa/

realclean: clean-kernels
	${MAKE} -C ${OCCA_DIR} clean

help:
	$(info $(value LIBP_HELP_MSG))
	@true

info:
	$(info LIBP_DIR  = $(LIBP_DIR))
	$(info OCCA_DIR  = $(OCCA_DIR))
	$(info LIBP_ARCH = $(LIBP_ARCH))
	$(info LIBP_CXXFLAGS  = $(LIBP_CXXFLAGS))
	@true
