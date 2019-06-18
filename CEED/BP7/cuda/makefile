cudaFlags =  -arch=sm_70   --use_fast_math -Xptxas -v,-dlcm=ca -Xcompiler=-g,-O0

BK7: BK7.cu
	nvcc -o BK7 $(cudaFlags) BK7.cu

clean:
	rm -f BK7

