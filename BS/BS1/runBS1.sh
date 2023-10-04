#!/bin/bash

function HELP {
  echo "Usage: ./runBS1.sh -m MODE"
  exit 1
}

#parse options
while getopts ":m:h:p:d:e" FLAG; do
  case "$FLAG" in
    m)
        mode=$OPTARG
        [[ ! $mode =~ CUDA|HIP|OpenCL|OpenMP|Serial|DPCPP ]] && {
            echo "Incorrect run mode provided"
            exit 1
        }
        ;;
    p)
        plat=$OPTARG;
	echo "platform=" $plat;;
    d)
        devi=$OPTARG;
	echo "device=" $devi;;
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

echo "Running BS1... on platform "$plat" and device "$devi" " 

#./BS1 -m $mode -b 1073741824
./BS1 -m ${mode} -bmin 1024 -bmax 1073741824 --bstep 1048576 -pl ${plat} -d ${devi}

#
# Noel Chalmers
# AMD Research
# 21/1/2020
#
