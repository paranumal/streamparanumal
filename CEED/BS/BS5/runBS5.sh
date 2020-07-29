#!/bin/bash

function HELP {
  echo "Usage: ./runBS5.sh -m MODE"
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
  cd ../CEED/BS/BS5
fi

export OCCA_DIR=${PWD}/../../../occa

make -j `nproc`

echo "Running BS5..."

#./BS5 -m $mode -b 1073741824
./BS5 -m $mode -bmin 1024 -bmax 1073741824  -nsamp 300

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
