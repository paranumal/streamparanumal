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

define LIB_HELP_MSG

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
	 Build core, mesh, ogs, and parAdogs libraries.
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

ifeq (,$(filter core mesh ogs primitives \
								clean clean-libs clean-kernels realclean info help, $(MAKECMDGOALS)))
ifneq (,$(MAKECMDGOALS))
$(error ${LIB_HELP_MSG})
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
LIBCORE_DIR=${LIBP_LIBS_DIR}/core
LIBMESH_DIR=${LIBP_LIBS_DIR}/mesh
LIBOGS_DIR=${LIBP_LIBS_DIR}/ogs
LIBPRIM_DIR=${HIPBONE_LIBS_DIR}/primitives

#includes
INCLUDES=${LIBP_INCLUDES}

#defines
LIBCORE_DEFINES=-DLIBP_DIR='"${LIBP_DIR}"'
LIBMESH_DEFINES=-DMESH_DIR='"${LIBMESH_DIR}"'
LIBOGS_DEFINES=-DLIBP_DIR='"${LIBP_DIR}"'  -DOGS_DIR='"${LIBOGS_DIR}"'
LIBPRIM_DEFINES=${LIBP_DEFINES}

ifeq (true,${gpu-aware-mpi})
  LIBOGS_DEFINES+= -DGPU_AWARE_MPI
endif

#.cpp compilation flags
LIB_CXXFLAGS=${LIBP_DEFINES} ${LIBP_CXXFLAGS} ${INCLUDES}

#object dependancies
LIB_DEPS=$(wildcard $(LIBP_INCLUDE_DIR)/*.h)  \
				 $(wildcard ${LIBP_INCLUDE_DIR}/*.hpp)

LIBOGS_DEPS=$(wildcard ${LIBP_INCLUDE_DIR}/ogs/*.hpp) \
						${LIB_DEPS}

LIBMESH_DEPS=$(wildcard ${LIBP_INCLUDE_DIR}/mesh/*.h)   \
						 $(wildcard ${LIBP_INCLUDE_DIR}/mesh/*.hpp) \
						 ${LIBPARADOGS_DEPS}

LIBCORE_SRC =$(wildcard core/*.cpp)
LIBMESH_SRC =$(wildcard mesh/*.cpp)
LIBOGS_SRC =$(wildcard ogs/*.cpp)
LIBPRIM_SRC =$(wildcard primitives/*.cpp)

LIBCORE_OBJS=$(LIBCORE_SRC:.cpp=.o)
LIBMESH_OBJS=$(LIBMESH_SRC:.cpp=.o)
LIBOGS_OBJS=$(LIBOGS_SRC:.cpp=.o)
LIBPRIM_OBJS=$(LIBPRIM_SRC:.cpp=.o)

.PHONY: all core prim mesh ogs \
				clean clean-kernels realclean silentUpdate-core  \
				silentUpdate-mesh


all: core prim mesh ogs

core: libcore.a silentUpdate-core
prim: libprim.a silentUpdate-prim
mesh: libmesh.a silentUpdate-mesh
ogs: libogs.a silentUpdate-ogs

libcore.a: $(LIBCORE_OBJS)
ifneq (,${verbose})
	ar -cr libcore.a $(LIBCORE_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr libcore.a $(LIBCORE_OBJS)
endif

libprim.a: $(LIBPRIM_OBJS) | libcore.a
ifneq (,${verbose})
	ar -cr libprim.a $(LIBPRIM_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr libprim.a $(LIBPRIM_OBJS)
endif

libogs.a: $(LIBOGS_OBJS) | libprim.a
ifneq (,${verbose})
	ar -cr libogs.a $(LIBOGS_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr libogs.a $(LIBOGS_OBJS)
endif

libmesh.a: $(LIBMESH_OBJS) | libogs.a
ifneq (,${verbose})
	ar -cr libmesh.a $(LIBMESH_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr libmesh.a $(LIBMESH_OBJS)
endif

silentUpdate-core:
	@true

silentUpdate-prim:
	@true

silentUpdate-ogs:
	@true

silentUpdate-mesh:
	@true

${OCCA_DIR}/lib/libocca.so:
	${MAKE} -C ${OCCA_DIR}

# rule for .cpp files
core/%.o: core/%.cpp $(LIB_DEPS) ${OCCA_DIR}/lib/libocca.so
ifneq (,${verbose})
	$(LIBP_CXX) -o $@ -c $< ${LIBCORE_DEFINES} $(LIB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(LIBP_CXX) -o $@ -c $< ${LIBCORE_DEFINES} $(LIB_CXXFLAGS)
endif

primitives/%.o: primitives/%.cpp $(LIB_DEPS) | libcore.a
ifneq (,${verbose})
	$(LIBP_CXX) -o $@ -c $< ${LIBPRIM_DEFINES} $(LIB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(LIBP_CXX) -o $@ -c $< ${LIBPRIM_DEFINES} $(LIB_CXXFLAGS)
endif

ogs/%.o: ogs/%.cpp $(LIBOGS_DEPS) | libprim.a
ifneq (,${verbose})
	$(LIBP_CXX) -o $@ -c $< ${LIBOGS_DEFINES} $(LIB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(LIBP_CXX) -o $@ -c $< ${LIBOGS_DEFINES} $(LIB_CXXFLAGS)
endif

mesh/%.o: mesh/%.cpp $(LIBMESH_DEPS) | libogs.a
ifneq (,${verbose})
	$(LIBP_CXX) -o $@ -c $< ${LIBMESH_DEFINES} $(LIB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(LIBP_CXX) -o $@ -c $< ${LIBMESH_DEFINES} $(LIB_CXXFLAGS)
endif

#cleanup
clean:
	rm -f core/*.o primitives/*.o  ogs/*.o mesh/*.o *.a

clean-kernels: clean
	rm -rf ${LIBP_DIR}/.occa

realclean: clean
	${MAKE} -C ${OCCA_DIR} clean

help:
	$(info $(value LIB_HELP_MSG))
	@true

info:
	$(info OCCA_DIR  = $(OCCA_DIR))
	$(info LIBP_DIR  = $(LIBP_DIR))
	$(info LIBP_ARCH = $(LIBP_ARCH))
	$(info CXXFLAGS  = $(LIB_CXXFLAGS))
	$(info LIBS      = $(LIBS))
	@true
