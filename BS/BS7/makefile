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

define BS7_HELP_MSG

Bakeoff Streaming 7 makefile targets:

   make BS7 (default)
   make clean
   make clean-libs
   make clean-kernels
   make realclean
   make info
   make help

Usage:

make BS7
   Build BS7 executable.
make clean
   Clean the BS7 executable and object files.
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

ifeq (,$(filter BS7 clean clean-libs clean-kernels \
                realclean info help,$(MAKECMDGOALS)))
ifneq (,$(MAKECMDGOALS))
$(error ${BS7_HELP_MSG})
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
BS7_LIBP_LIBS=mesh ogs core

#includes
INCLUDES=${LIBP_INCLUDES} \
         -I.

#defines
DEFINES =${LIBP_DEFINES} \
         -DSTREAM_DIR='"${STREAM_DIR}"'

#.cpp compilation flags
BS7_CXXFLAGS=${LIBP_CXXFLAGS} ${DEFINES} ${INCLUDES}

#link libraries
LIBS=-L${LIBP_LIBS_DIR} $(addprefix -l,$(BS7_LIBP_LIBS)) \
     ${LIBP_LIBS}

#link flags
LFLAGS=${BS7_CXXFLAGS} ${LIBS}

#object dependancies
DEPS=$(wildcard *.hpp) \
     $(wildcard $(LIBP_INCLUDE_DIR)/*.h) \
     $(wildcard $(LIBP_INCLUDE_DIR)/*.hpp)

SRC =$(wildcard src/*.cpp)

OBJS=$(SRC:.cpp=.o)

.PHONY: all libp_libs clean clean-libs \
		clean-kernels realclean help info BS7

all: BS7

${OCCA_DIR}/lib/libocca.so:
	${MAKE} -C ${OCCA_DIR}

libp_libs: ${OCCA_DIR}/lib/libocca.so
ifneq (,${verbose})
	${MAKE} -C ${LIBP_LIBS_DIR} $(BS7_LIBP_LIBS) verbose=${verbose}
else
	@${MAKE} -C ${LIBP_LIBS_DIR} $(BS7_LIBP_LIBS) --no-print-directory
endif

BS7:$(OBJS) bs7Main.o libp_libs
ifneq (,${verbose})
	$(LIBP_LD) -o BS7 bs7Main.o $(OBJS) $(MESH_OBJS) $(LFLAGS)
else
	@printf "%b" "$(EXE_COLOR)Linking $(@F)$(NO_COLOR)\n";
	@$(LIBP_LD) -o BS7 bs7Main.o $(OBJS) $(MESH_OBJS) $(LFLAGS)
endif

# rule for .cpp files
%.o: %.cpp $(DEPS) | libp_libs
ifneq (,${verbose})
	$(LIBP_CXX) -o $*.o -c $*.cpp $(BS7_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(LIBP_CXX) -o $*.o -c $*.cpp $(BS7_CXXFLAGS)
endif

#cleanup
clean:
	rm -f src/*.o *.o BS7

clean-libs: clean
	${MAKE} -C ${LIBP_LIBS_DIR} clean

clean-kernels: clean-libs
	rm -rf ${STREAM_DIR}/.occa/

realclean: clean-kernels
	${MAKE} -C ${OCCA_DIR} clean

help:
	$(info $(value BS7_HELP_MSG))
	@true

info:
	$(info STREAM_DIR  = $(STREAM_DIR))
	$(info OCCA_DIR  = $(OCCA_DIR))
	$(info LIBP_ARCH = $(LIBP_ARCH))
	$(info LIBP_CXXFLAGS  = $(BS7_CXXFLAGS))
	$(info LIBS      = $(LIBS))
	@true
