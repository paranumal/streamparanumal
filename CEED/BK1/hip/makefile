hipFlags = --amdgpu-target=gfx906 -O3
#-ffast-math 

BK1: BK1.cpp makefile
	hipcc -o BK1 $(hipFlags) BK1.cpp

clean:
	rm -f BK1
