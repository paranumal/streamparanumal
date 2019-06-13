cudaFlags = -arch=sm_70   --use_fast_math -Xptxas -v,-dlcm=ca,-O3 -Xcompiler -O3

BK1: BK1.cu makefile
	nvcc -o BK1 $(cudaFlags) BK1.cu

clean:
	rm -f BK1 BK1.o

