
PROJ_DIR=./

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

${PROJ_DIR}/BK6: ${PROJ_DIR}/BK6.cpp makefile
	$(compiler) -DOCCA_VERSION_1_0 $(compilerFlags) -o BK6 $(linkerFlags) ${PROJ_DIR}/BK6.cpp ${PROJ_DIR}/meshBasis.cpp $(paths) -L${OCCA_DIR}/lib $(flags) -locca -llapack -lblas
#$(linkerFlags)


$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) -DOCCA_VERSION_1_0 $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f ${PROJ_DIR}/BK1

