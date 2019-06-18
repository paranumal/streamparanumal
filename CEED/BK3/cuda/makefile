cudaFlags =  -arch=sm_70   --use_fast_math -Xptxas -v,-dlcm=ca,-O3 -Xcompiler -O3

flags = -Ddfloat=double
flags += -DdfloatString='"double"'
flags += -I../../core

BK3: BK3.cu
	mpic++ $(flags) -c  -I../../core/ ../../core/meshBasis.cpp
	nvcc -o BK3 $(cudaFlags) BK3.cu meshBasis.o -llapack -lblas

clean:
	rm -f BK3 meshBasis.o

