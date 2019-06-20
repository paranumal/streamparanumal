
occa:
	cd BK1/occa; make 
	cd BK2/occa; make 
	cd BK3/occa; make 
	cd BK4/occa; make 
	cd BK5/occa; make 
	cd BK6/occa; make 
	cd BK9/occa; make 

cuda:
	cd BK1/cuda; make 
	cd BK3/cuda; make
	cd BK5/cuda; make 


hip:
	cd BK1/hip; make 
	cd BK3/hip; make
	cd BK5/hip; make 

all: cuda hip occa

cudaclean:
	cd BK1/cuda; make clean
	cd BK3/cuda; make clean
	cd BK5/cuda; make clean

hipclean:
	cd BK1/hip; make clean
	cd BK3/hip; make clean
	cd BK5/hip; make clean

occaclean:
	cd BK1/occa; make clean
	cd BK2/occa; make clean
	cd BK3/occa; make clean
	cd BK4/occa; make clean
	cd BK5/occa; make clean
	cd BK6/occa; make clean
	cd BK9/occa; make clean

clean: cudaclean hipclean occaclean
