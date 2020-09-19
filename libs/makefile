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

Libraries makefile targets:

	 make all (default)
	 make lib{}
	 make clean
	 make clean-kernels
	 make realclean
	 make info
	 make help

Usage:

make all
	 Build core, ogs, mesh, and linAlg libraries.
make {lib}
	 Build only {lib} library.
make clean
	 Clean all libraries, and object files.
make clean-kernels
	 In addition to "make clean", also cleans the cached OCCA kernels.
make realclean
	 In addition to "make clean-kernels", also clean 3rd party libraries.
make info
	 List directories and compiler flags in use.
make help
	 Display this help message.

Can use "make verbose=true" for verbose output.

endef

ifeq (,$(filter core linAlg mesh ogs \
								clean clean-libs clean-kernels realclean info help, $(MAKECMDGOALS)))
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

#third party
GS_DIR=${CEED_TPL_DIR}/gslib

#libraries
LIBCORE_DIR=${CEED_LIBS_DIR}/core
LIBLINALG_DIR=${CEED_LIBS_DIR}/linAlg
LIBMESH_DIR=${CEED_LIBS_DIR}/mesh
LIBOGS_DIR=${CEED_LIBS_DIR}/ogs

#includes
INCLUDES=${CEED_INCLUDES}

#defines
LIBCORE_DEFINES=
LIBLINALG_DEFINES=-DLINALG_DIR='"${LIBLINALG_DIR}"'
LIBMESH_DEFINES=-DMESH_DIR='"${LIBMESH_DIR}"'
LIBOGS_DEFINES=-DCEED_DIR='"${CEED_DIR}"'  -DOGS_DIR='"${LIBOGS_DIR}"'

#.cpp compilation flags
LIB_CXXFLAGS=${CEED_DEFINES} ${CEED_MPICXXFLAGS} ${INCLUDES}

#object dependancies
LIB_DEPS=$(wildcard $(CEED_INCLUDE_DIR)/*.h)  \
				 $(wildcard ${CEED_INCLUDE_DIR}/*.hpp)

LIBMESH_DEPS=$(wildcard ${CEED_INCLUDE_DIR}/mesh/*.h)   \
						 $(wildcard ${CEED_INCLUDE_DIR}/mesh/*.hpp) \
						 ${LIB_DEPS}

LIBOGS_DEPS=$(wildcard ${CEED_INCLUDE_DIR}/ogs/*.h)   \
						$(wildcard ${CEED_INCLUDE_DIR}/ogs/*.hpp) \
						${LIB_DEPS}

LIBCORE_SRC =$(wildcard core/*.cpp)
LIBLINALG_SRC =$(wildcard linAlg/*.cpp)
LIBMESH_SRC =$(wildcard mesh/*.cpp)
LIBOGS_SRC =$(wildcard ogs/*.cpp)

LIBCORE_OBJS=$(LIBCORE_SRC:.cpp=.o)
LIBLINALG_OBJS=$(LIBLINALG_SRC:.cpp=.o)
LIBMESH_OBJS=$(LIBMESH_SRC:.cpp=.o)
LIBOGS_OBJS=$(LIBOGS_SRC:.cpp=.o)

.PHONY: all core linAlg mesh ogs  \
				clean realclean silentUpdate-core             \
				silentUpdate-linAlg silentUpdate-ogs silentUpdate-mesh

all: core linAlg mesh ogs

core: libcore.a silentUpdate-core
linAlg: liblinAlg.a silentUpdate-linAlg
mesh: libmesh.a silentUpdate-mesh
ogs: libogs.a silentUpdate-ogs

libcore.a: $(LIBCORE_OBJS)
ifneq (,${verbose})
	ar -cr libcore.a $(LIBCORE_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr libcore.a $(LIBCORE_OBJS)
endif

libgs: | libcore.a
ifneq (,${verbose})
	${MAKE} -C $(GS_DIR) install verbose=${verbose}
else
	@${MAKE} -C $(GS_DIR) install --no-print-directory
endif

libogs.a: $(LIBOGS_OBJS) | libgs
ifneq (,${verbose})
	ar -cr libogs.a $(LIBOGS_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr libogs.a $(LIBOGS_OBJS)
endif

liblinAlg.a: $(LIBLINALG_OBJS)
ifneq (,${verbose})
	ar -cr liblinAlg.a $(LIBLINALG_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr liblinAlg.a $(LIBLINALG_OBJS)
endif

libmesh.a: $(LIBMESH_OBJS) | liblinAlg.a libogs.a
ifneq (,${verbose})
	ar -cr libmesh.a $(LIBMESH_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr libmesh.a $(LIBMESH_OBJS)
endif

silentUpdate-core:
	@true

silentUpdate-linAlg:
	@true

silentUpdate-ogs:
	@true

silentUpdate-mesh:
	@true

# rule for .cpp files
core/%.o: core/%.cpp $(LIB_DEPS)
ifneq (,${verbose})
	$(CEED_MPICXX) -o $@ -c $< ${LIBCORE_DEFINES} $(LIB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(CEED_MPICXX) -o $@ -c $< ${LIBCORE_DEFINES} $(LIB_CXXFLAGS)
endif

linAlg/%.o: linAlg/%.cpp $(LIB_DEPS) | libcore.a
ifneq (,${verbose})
	$(CEED_MPICXX) -o $@ -c $< ${LIBLINALG_DEFINES} $(LIB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(CEED_MPICXX) -o $@ -c $< ${LIBLINALG_DEFINES} $(LIB_CXXFLAGS)
endif

mesh/%.o: mesh/%.cpp $(LIBMESH_DEPS) | libogs.a
ifneq (,${verbose})
	$(CEED_MPICXX) -o $@ -c $< ${LIBMESH_DEFINES} $(LIB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(CEED_MPICXX) -o $@ -c $< ${LIBMESH_DEFINES} $(LIB_CXXFLAGS)
endif

ogs/%.o: ogs/%.cpp $(LIBOGS_DEPS) | libgs
ifneq (,${verbose})
	$(CEED_MPICXX) -o $@ -c $< ${LIBOGS_DEFINES} -I${GS_DIR}/src $(LIB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(CEED_MPICXX) -o $@ -c $< ${LIBOGS_DEFINES} -I${GS_DIR}/src $(LIB_CXXFLAGS)
endif

#cleanup
clean:
	rm -f core/*.o linAlg/*.o mesh/*.o ogs/*.o *.a

clean-kernels: clean
#   $(shell ${OCCA_DIR}/bin/occa clear all -y)
	rm -rf ~/.occa/

realclean: clean
	${MAKE} -C ${GS_DIR} clean

help:
	$(info $(value CEED_HELP_MSG))
	@true

info:
	$(info OCCA_DIR  = $(OCCA_DIR))
	$(info CEED_DIR  = $(CEED_DIR))
	$(info CEED_ARCH = $(CEED_ARCH))
	$(info CXXFLAGS  = $(LIB_CXXFLAGS))
	$(info LIBS      = $(LIBS))
	@true