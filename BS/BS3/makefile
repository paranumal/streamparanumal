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

define BS3_HELP_MSG

Bakeoff Streaming 3 makefile targets:

   make BS3 (default)
   make clean
   make clean-libs
   make clean-kernels
   make realclean
   make info
   make help

Usage:

make BS3
   Build BS3 executable.
make clean
   Clean the BS3 executable and object files.
make clean-libs
   In addition to "make clean", also clean the mesh, ogs, and core libraries.
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

ifeq (,$(filter BS3 clean clean-libs clean-kernels \
                realclean info help,$(MAKECMDGOALS)))
ifneq (,$(MAKECMDGOALS))
$(error ${BS3_HELP_MSG})
endif
endif

ifndef LIBP_MAKETOP_LOADED
ifeq (,$(wildcard ../../make.top))
$(error cannot locate ${PWD}/../../make.top)
else
include ../../make.top
endif
endif

#libraries
BS3_LIBP_LIBS=core

#includes
INCLUDES=${LIBP_INCLUDES} \
         -I.

#defines
DEFINES =${LIBP_DEFINES} \
         -DSTREAM_DIR='"${STREAM_DIR}"'

#.cpp compilation flags
BS3_CXXFLAGS=${LIBP_CXXFLAGS} ${DEFINES} ${INCLUDES}

#link libraries
LIBS=-L${LIBP_LIBS_DIR} $(addprefix -l,$(BS3_LIBP_LIBS)) \
     ${LIBP_LIBS}

#link flags
LFLAGS=${BS3_CXXFLAGS} ${LIBS}

#object dependancies
DEPS=$(wildcard *.hpp) \
     $(wildcard $(LIBP_INCLUDE_DIR)/*.h) \
     $(wildcard $(LIBP_INCLUDE_DIR)/*.hpp)

SRC =$(wildcard src/*.cpp)

OBJS=$(SRC:.cpp=.o)

.PHONY: all libp_libs clean clean-libs \
		clean-kernels realclean help info BS3

all: BS3

libp_libs: ${OCCA_DIR}/lib/libocca.so
ifneq (,${verbose})
	${MAKE} -C ${LIBP_LIBS_DIR} $(BS3_LIBP_LIBS) verbose=${verbose}
else
	@${MAKE} -C ${LIBP_LIBS_DIR} $(BS3_LIBP_LIBS) --no-print-directory
endif

BS3:$(OBJS) bs3Main.o libp_libs
ifneq (,${verbose})
	$(LIBP_LD) -o BS3 bs3Main.o $(OBJS) $(MESH_OBJS) $(LFLAGS)
else
	@printf "%b" "$(EXE_COLOR)Linking $(@F)$(NO_COLOR)\n";
	@$(LIBP_LD) -o BS3 bs3Main.o $(OBJS) $(MESH_OBJS) $(LFLAGS)
endif

${OCCA_DIR}/lib/libocca.so:
	${MAKE} -C ${OCCA_DIR}

# rule for .cpp files
%.o: %.cpp $(DEPS) | libp_libs
ifneq (,${verbose})
	$(LIBP_CXX) -o $*.o -c $*.cpp $(BS3_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(LIBP_CXX) -o $*.o -c $*.cpp $(BS3_CXXFLAGS)
endif

#cleanup
clean:
	rm -f src/*.o *.o BS3

clean-libs: clean
	${MAKE} -C ${LIBP_LIBS_DIR} clean

clean-kernels: clean-libs
	rm -rf ${STREAM_DIR}/.occa/

realclean: clean-kernels
	${MAKE} -C ${OCCA_DIR} clean

help:
	$(info $(value BS3_HELP_MSG))
	@true

info:
	$(info STREAM_DIR  = $(STREAM_DIR))
	$(info OCCA_DIR  = $(OCCA_DIR))
	$(info LIBP_ARCH = $(LIBP_ARCH))
	$(info CXXFLAGS  = $(BS3_CXXFLAGS))
	$(info LIBS      = $(LIBS))
	@true
