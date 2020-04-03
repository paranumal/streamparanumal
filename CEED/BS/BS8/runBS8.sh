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
  cd ../CEED/BS/BS8
fi

export OCCA_DIR=${PWD}/../../../occa

make -j `nproc`

echo "Running BS8..."

./BS8 -m $mode -nx 160 -ny 80 -nz 80 -p 1
./BS8 -m $mode -nx  80 -ny 80 -nz 40 -p 2
./BS8 -m $mode -nx  80 -ny 40 -nz 40 -p 3
./BS8 -m $mode -nx  40 -ny 40 -nz 40 -p 4
./BS8 -m $mode -nx  40 -ny 40 -nz 20 -p 5
./BS8 -m $mode -nx  40 -ny 20 -nz 20 -p 6
./BS8 -m $mode -nx  30 -ny 20 -nz 20 -p 7
./BS8 -m $mode -nx  20 -ny 20 -nz 20 -p 8

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
