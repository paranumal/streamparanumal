cudaFlags =  -arch=sm_70   --use_fast_math -Xptxas -v,-dlcm=ca,-O3 -Xcompiler -O3

flags = -Ddfloat=double
flags += -DdfloatString='"double"'
flags += -I../../core

BK5: BK5.cu
	mpic++ $(flags) -c  -I../../core/ ../../core/meshBasis.cpp
	nvcc $(flags) -o BK5 $(cudaFlags) BK5.cu  meshBasis.o -llapack -lblas

clean:
	rm -f BK5 meshBasis.o

