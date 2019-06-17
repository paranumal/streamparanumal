
PROJ_DIR=./

ifndef OCCA_DIR
ERROR:
	@echo "ERROR:  Environment variable OCCA_DIR is not set."
endif

include ${OCCA_DIR}/scripts/Makefile

compiler = mpic++
compilerFlags += -Ddfloat=double
compilerFlags += -Ddlong=int
compilerFlags += -DdfloatString='"double"'
compilerFlags += -DdlongString='"int"'

${PROJ_DIR}/BK5: ${PROJ_DIR}/BK5.cpp
	$(compiler) -DOCCA_VERSION_1_0 $(compilerFlags) -o BK5 $(linkerFlags) ${PROJ_DIR}/BK5.cpp ${PROJ_DIR}/meshBasis.cpp $(paths) -L${OCCA_DIR}/lib $(flags) -locca -llapack -lblas
#$(linkerFlags)


$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) -DOCCA_VERSION_1_0 $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(oPath)/*;
	rm -f ${PROJ_DIR}/BK1

