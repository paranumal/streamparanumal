
occa:
	cd BK/BK1/occa; make 
	cd BK/BK2/occa; make 
	cd BK/BK3/occa; make 
	cd BK/BK4/occa; make 
	cd BK/BK5/occa; make 
	cd BK/BK6/occa; make 
#	cd BK/BK9/occa; make 
	cd BP/occa; make

cuda:
	cd BK/BK1/cuda; make 
	cd BK/BK3/cuda; make
	cd BK/BK5/cuda; make 


hip:
	cd BK/BK1/hip; make 
	cd BK/BK3/hip; make
	cd BK/BK5/hip; make 

all: cuda hip occa

cudaclean:
	cd BK/BK1/cuda; make clean
	cd BK/BK3/cuda; make clean
	cd BK/BK5/cuda; make clean

hipclean:
	cd BK/BK1/hip; make clean
	cd BK/BK3/hip; make clean
	cd BK/BK5/hip; make clean

occaclean:
	cd BK/BK1/occa; make clean
	cd BK/BK2/occa; make clean
	cd BK/BK3/occa; make clean
	cd BK/BK4/occa; make clean
	cd BK/BK5/occa; make clean
	cd BK/BK6/occa; make clean
#	cd BK/BK9/occa; make clean
	cd BP/occa; make clean

clean: cudaclean hipclean occaclean
