#!/bin/bash

function HELP {
  echo "Usage: ./runBS1.sh -m MODE"
  exit 1
}

#parse options
while getopts :m:h FLAG; do
  case $FLAG in
    m)
        mode=$OPTARG
        [[ ! $mode =~ CUDA|HIP|OpenCL|OpenMP|Serial|DPCPP ]] && {
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

# Build the code
# make -j `nproc`

if [ -z $mode ]; then
    echo "No mode supplied, defaulting to HIP"
    mode=HIP
fi

echo "Running BS1..."

#./BS1 -m $mode -b 1073741824
./BS1 -m $mode -bmin 1024 -bmax 1073741824 --bstep 1048576 -pl 4 -d 0

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
