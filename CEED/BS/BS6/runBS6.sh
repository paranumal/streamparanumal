#!/bin/bash

function HELP {
  echo "Usage: ./runBS6.sh -m MODE"
  exit 1
}

#parse options
while getopts :m:h FLAG; do
  case $FLAG in
    m)
        mode=$OPTARG
        [[ ! $mode =~ CUDA|HIP|OpenCL|OpenMP|Serial ]] && {
            echo "Incorrect run mode provided"
            exit 1
        }
        ;;
    h)  #show help
        HELP
        ;;
    \?) #unrecognized option - show help
        HELP
        ;;
  esac
done

if [ -z $mode ]
then
    echo "No mode supplied, defaulting to HIP"
    mode=HIP
fi

# Build the code

if [ ! -d "../../../occa" ]; then
  cd ../../../
  git clone https://github.com/libocca/occa
  cd occa; make -j `nproc`
  cd ../CEED/BS/BS6
fi

export OCCA_DIR=${PWD}/../../../occa

make -j `nproc`

echo "Running BS6..."

for n in `seq 8 2 120`
do
    ./BS6 -m $mode -nx $n -ny $n -nz $n -p 1
done

for n in `seq 4 2 100`
do
    ./BS6 -m $mode -nx $n -ny $n -nz $n -p 2
done

for n in `seq 2 2 80`
do
    ./BS6 -m $mode -nx $n -ny $n -nz $n -p 3
done

for n in `seq 2 2 60`
do
    ./BS6 -m $mode -nx $n -ny $n -nz $n -p 4
done


for n in `seq 2 2 50`
do
    ./BS6 -m $mode -nx $n -ny $n -nz $n -p 5
done


for n in `seq 2 32`
do
    ./BS6 -m $mode -nx $n -ny $n -nz $n -p 6
done


for n in `seq 4 24`
do
    ./BS6 -m $mode -nx $n -ny $n -nz $n -p 7
done


#mpirun -np 1 BS6 -m $mode -nx  80 -ny  80 -nz  80 -p 2
#mpirun -np 1 BS6 -m $mode -nx  53 -ny  53 -nz  53 -p 3
#mpirun -np 1 BS6 -m $mode -nx  40 -ny  40 -nz  40 -p 4
#mpirun -np 1 BS6 -m $mode -nx  32 -ny  32 -nz  32 -p 5
#mpirun -np 1 BS6 -m $mode -nx  27 -ny  27 -nz  27 -p 6
#mpirun -np 1 BS6 -m $mode -nx  23 -ny  23 -nz  23 -p 7
#mpirun -np 1 BS6 -m $mode -nx  20 -ny  20 -nz  20 -p 8

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
