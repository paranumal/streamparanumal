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

define BS1_HELP_MSG

Bakeoff Streaming 1 makefile targets:

   make BS1 (default)
   make clean
   make clean-libs
   make clean-kernels
   make realclean
   make info
   make help

Usage:

make BS1
   Build BS1 executable.
make clean
   Clean the BS1 executable and object files.
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

ifeq (,$(filter BS1 clean clean-libs clean-kernels \
                realclean info help,$(MAKECMDGOALS)))
ifneq (,$(MAKECMDGOALS))
$(error ${BS1_HELP_MSG})
endif
endif

ifndef CEED_MAKETOP_LOADED
ifeq (,$(wildcard ../../make.top))
$(error cannot locate ${PWD}/../../make.top)
else
include ../../make.top
endif
endif

#libraries
BS1_CEED_LIBS=linAlg core

#includes
INCLUDES=${CEED_INCLUDES} \
         -I.

#defines
DEFINES =${CEED_DEFINES} \
         -DCEED_DIR='"${CEED_DIR}"'

#.cpp compilation flags
BS1_CXXFLAGS=${CEED_CXXFLAGS} ${DEFINES} ${INCLUDES}

#link libraries
LIBS=-L${CEED_LIBS_DIR} $(addprefix -l,$(BS1_CEED_LIBS)) \
     ${CEED_LIBS}

#link flags
LFLAGS=${BS1_CXXFLAGS} ${LIBS}

#object dependancies
DEPS=$(wildcard *.hpp) \
     $(wildcard $(CEED_INCLUDE_DIR)/*.h) \
     $(wildcard $(CEED_INCLUDE_DIR)/*.hpp)

SRC =$(wildcard src/*.cpp)

OBJS=$(SRC:.cpp=.o)

.PHONY: all ceed_libs clean clean-libs \
		clean-kernels realclean help info BS1

all: BS1

ceed_libs:
ifneq (,${verbose})
	${MAKE} -C ${CEED_LIBS_DIR} $(BS1_CEED_LIBS) verbose=${verbose}
else
	@${MAKE} -C ${CEED_LIBS_DIR} $(BS1_CEED_LIBS) --no-print-directory
endif

BS1:$(OBJS) bs1Main.o ceed_libs
ifneq (,${verbose})
	$(CEED_LD) -o BS1 bs1Main.o $(OBJS) $(MESH_OBJS) $(LFLAGS)
else
	@printf "%b" "$(EXE_COLOR)Linking $(@F)$(NO_COLOR)\n";
	@$(CEED_LD) -o BS1 bs1Main.o $(OBJS) $(MESH_OBJS) $(LFLAGS)
endif

# rule for .cpp files
%.o: %.cpp $(DEPS) | ceed_libs
ifneq (,${verbose})
	$(CEED_MPICXX) -o $*.o -c $*.cpp $(BS1_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(CEED_MPICXX) -o $*.o -c $*.cpp $(BS1_CXXFLAGS)
endif

#cleanup
clean:
	rm -f src/*.o *.o BS1

clean-libs: clean
	${MAKE} -C ${CEED_LIBS_DIR} clean

clean-kernels: clean-libs
	rm -rf ~/.occa/

realclean: clean
	${MAKE} -C ${CEED_LIBS_DIR} realclean

help:
	$(info $(value BS1_HELP_MSG))
	@true

info:
	$(info OCCA_DIR  = $(OCCA_DIR))
	$(info CEED_DIR  = $(CEED_DIR))
	$(info CEED_ARCH = $(CEED_ARCH))
	$(info CXXFLAGS  = $(BS1_CXXFLAGS))
	$(info LIBS      = $(LIBS))
	@true
