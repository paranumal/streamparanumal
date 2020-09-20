#!/bin/bash

function HELP {
  echo "Usage: ./runBS8.sh -m MODE"
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

if [ -z "$OCCA_DIR" ]; then
  echo "Error: OCCA_DIR not set."
  exit 2
fi

# Build the code
make -j `nproc`

if [ -z $mode ]; then
    echo "No mode supplied, defaulting to HIP"
    mode=HIP
fi

echo "Running BS8..."


for n in `seq 8 2 120`
do
    ./BS8 -m $mode -nx $n -ny $n -nz $n -p 1
done

for n in `seq 4 2 100`
do
    ./BS8 -m $mode -nx $n -ny $n -nz $n -p 2
done

for n in `seq 2 2 80`
do
    ./BS8 -m $mode -nx $n -ny $n -nz $n -p 3
done

for n in `seq 2 2 60`
do
    ./BS8 -m $mode -nx $n -ny $n -nz $n -p 4
done


for n in `seq 2 2 50`
do
    ./BS8 -m $mode -nx $n -ny $n -nz $n -p 5
done


for n in `seq 2 32`
do
    ./BS8 -m $mode -nx $n -ny $n -nz $n -p 6
done


for n in `seq 4 24`
do
    ./BS8 -m $mode -nx $n -ny $n -nz $n -p 7
done

# mpirun -np 1 BS8 -m $mode -nx 159 -ny 159 -nz 159 -p 1
# mpirun -np 1 BS8 -m $mode -nx  80 -ny  80 -nz  80 -p 2
# mpirun -np 1 BS8 -m $mode -nx  53 -ny  53 -nz  53 -p 3
# mpirun -np 1 BS8 -m $mode -nx  40 -ny  40 -nz  40 -p 4
# mpirun -np 1 BS8 -m $mode -nx  32 -ny  32 -nz  32 -p 5
# mpirun -np 1 BS8 -m $mode -nx  27 -ny  27 -nz  27 -p 6
# mpirun -np 1 BS8 -m $mode -nx  23 -ny  23 -nz  23 -p 7
# mpirun -np 1 BS8 -m $mode -nx  20 -ny  20 -nz  20 -p 8

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
