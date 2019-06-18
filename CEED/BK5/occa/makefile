
PROJ_DIR=./
CORE_DIR=../../core

ifndef OCCA_DIR
ERROR:
	@echo "ERROR:  Environment variable OCCA_DIR is not set."
endif

include ${OCCA_DIR}/scripts/Makefile

compiler = mpic++

compilerFlags += -Ddlong=int
compilerFlags += -DdlongString='"int"'
compilerFlags += -Ddfloat=double
compilerFlags += -DdfloatString='"double"'

${PROJ_DIR}/BK5: ${PROJ_DIR}/BK5.cpp makefile ${CORE_DIR}/meshBasis.cpp
	$(compiler) -DOCCA_VERSION_1_0 $(compilerFlags) -o BK5 $(linkerFlags) ${PROJ_DIR}/BK5.cpp -I${CORE_DIR}/ ${CORE_DIR}/meshBasis.cpp $(paths) -L${OCCA_DIR}/lib $(flags) -locca -llapack -lblas
#$(linkerFlags)


$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) -DOCCA_VERSION_1_0 $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f ${PROJ_DIR}/BK5

