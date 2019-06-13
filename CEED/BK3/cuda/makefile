cudaFlags =  -arch=sm_70   --use_fast_math -Xptxas -v,-dlcm=ca,-O3 -Xcompiler -O3

BK3: BK3.cu
	nvcc -o BK3 $(cudaFlags) BK3.cu

clean:
	rm -f BK3

