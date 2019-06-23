hipFlags = --amdgpu-target=gfx906  -O3 -ffp-contract=fast
#ffp-contract=fast

flags = -Ddfloat=double
flags += -DdfloatString='"double"'
flags += -I../../core

BK5: BK5.cpp makefile
	mpic++ $(flags) -c  -I../../core/ ../../core/meshBasis.cpp
	hipcc $(flags) -o BK5 $(hipFlags) BK5.cpp meshBasis.o -llapack -lblas

clean:
	rm -f BK5 meshBasis.o
