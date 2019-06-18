hipFlags = --amdgpu-target=gfx906 -O3
#-ffast-math 

flags = -Ddfloat=double
flags += -DdfloatString='"double"'
flags += -I../../core


BK1: BK1.cpp makefile
	mpic++ $(flags) -c  -I../../core/ ../../core/meshBasis.cpp
	hipcc -o BK1 $(hipFlags) BK1.cpp meshBasis.o -llapack -lblas

clean:
	rm -f BK1
