cudaFlags =  -arch=sm_70   --use_fast_math -Xptxas -v,-dlcm=ca,-O3 -Xcompiler -g

BK5: BK5.cu
	nvcc -o BK5 $(cudaFlags) BK5.cu

clean:
	rm -f BK5

