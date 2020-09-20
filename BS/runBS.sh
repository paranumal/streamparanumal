#!/bin/bash

function HELP {
  echo "Usage: ./runBS.sh -m MODE"
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

for bs in `seq 0 8`
do
BS${bs}/BS${bs} -m $mode
done

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
